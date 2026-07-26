package providers_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/omarluq/cc-relay/internal/providers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testOpenAIProviderName = "openai"

func testOpenAIConfig() *providers.OpenAIConfig {
	return &providers.OpenAIConfig{
		ModelMapping:    nil,
		Name:            testOpenAIProviderName,
		BaseURL:         "",
		ReasoningEffort: "",
		Models:          nil,
	}
}

func TestOpenAIProviderDefaultsAndAuthentication(t *testing.T) {
	t.Parallel()

	provider := providers.NewOpenAIProvider(testOpenAIConfig())

	assert.Equal(t, providers.DefaultOpenAIBaseURL, provider.BaseURL())
	assert.Equal(t, providers.OpenAIOwner, provider.Owner())
	require.Len(t, provider.ListModels(), 1)
	assert.Equal(t, "gpt-5.6-sol", provider.ListModels()[0].ID)
	assert.True(t, provider.RequiresBodyTransform())
	assert.False(t, provider.SupportsTransparentAuth())

	req, err := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		"https://api.openai.com/v1/responses",
		http.NoBody,
	)
	require.NoError(t, err)
	require.NoError(t, provider.Authenticate(req, "sk-test"))
	assert.Equal(t, "Bearer sk-test", req.Header.Get("Authorization"))
	assert.Empty(t, req.Header.Get("x-api-key"))
}

func TestOpenAIProviderTransformsAnthropicRequest(t *testing.T) {
	t.Parallel()

	provider := providers.NewOpenAIProvider(&providers.OpenAIConfig{
		Name:            "openai",
		BaseURL:         "https://openai.example/v1/",
		ReasoningEffort: "high",
		ModelMapping: map[string]string{
			"gpt": "gpt-5.6-sol",
		},
		Models: nil,
	})

	source := `{
		"model":"gpt",
		"max_tokens":4096,
		"stream":false,
		"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}],
		"tools":[{
			"name":"read_file",
			"description":"Read a file",
			"input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
		}],
		"tool_choice":{"type":"tool","name":"read_file","disable_parallel_tool_use":true},
		"messages":[
			{"role":"user","content":[{"type":"text","text":"Read README.md"}]},
			{"role":"assistant","content":[
				{"type":"text","text":"I'll inspect it."},
				{"type":"thinking","thinking":"private"},
				{"type":"tool_use","id":"toolu_123","name":"read_file","input":{"path":"README.md"}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"toolu_123","content":"hello"}
			]}
		]
	}`

	transformed, targetURL, err := provider.TransformRequest([]byte(source), "/v1/messages")
	require.NoError(t, err)
	assert.Equal(t, "https://openai.example/v1/responses", targetURL)

	var request map[string]any
	require.NoError(t, json.Unmarshal(transformed, &request))
	assert.Equal(t, "gpt-5.6-sol", request["model"])
	assert.Equal(t, "You are a coding agent.", request["instructions"])
	assert.Equal(t, float64(4096), request["max_output_tokens"])
	assert.Equal(t, false, request["store"])
	assert.Equal(t, false, request["parallel_tool_calls"])
	assert.Equal(t, map[string]any{"effort": "high"}, request["reasoning"])
	assert.Equal(t, map[string]any{"name": "read_file", "type": "function"}, request["tool_choice"])

	input, inputIsSlice := request["input"].([]any)
	require.True(t, inputIsSlice)
	require.Len(t, input, 4)
	firstInput, firstInputIsMap := input[0].(map[string]any)
	require.True(t, firstInputIsMap)
	toolCallInput, toolCallInputIsMap := input[2].(map[string]any)
	require.True(t, toolCallInputIsMap)
	toolResultInput, toolResultInputIsMap := input[3].(map[string]any)
	require.True(t, toolResultInputIsMap)
	assert.Equal(t, "message", firstInput["type"])
	assert.Equal(t, "function_call", toolCallInput["type"])
	assert.Equal(t, "toolu_123", toolCallInput["call_id"])
	assert.Equal(t, "function_call_output", toolResultInput["type"])
	assert.Equal(t, "hello", toolResultInput["output"])

	tools, toolsIsSlice := request["tools"].([]any)
	require.True(t, toolsIsSlice)
	require.Len(t, tools, 1)
	firstTool, firstToolIsMap := tools[0].(map[string]any)
	require.True(t, firstToolIsMap)
	assert.Equal(t, "function", firstTool["type"])
	assert.Equal(t, false, firstTool["strict"])
}

func TestOpenAIProviderTransformsNonStreamingResponse(t *testing.T) {
	t.Parallel()

	provider := providers.NewOpenAIProvider(testOpenAIConfig())
	source := `{
		"id":"resp_123",
		"model":"gpt-5.6-sol",
		"status":"completed",
		"output":[
			{"type":"reasoning","summary":[]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I'll read it."}]},
			{"type":"function_call","call_id":"call_123","name":"read_file","arguments":"{\"path\":\"README.md\"}"}
		],
		"usage":{"input_tokens":42,"output_tokens":17}
	}`
	resp := &http.Response{
		StatusCode:    http.StatusOK,
		Header:        http.Header{"Content-Type": []string{"application/json"}},
		Body:          io.NopCloser(strings.NewReader(source)),
		ContentLength: int64(len(source)),
	}

	require.NoError(t, provider.TransformHTTPResponse(resp))
	transformed, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	var message map[string]any
	require.NoError(t, json.Unmarshal(transformed, &message))
	assert.Equal(t, "resp_123", message["id"])
	assert.Equal(t, "message", message["type"])
	assert.Equal(t, "tool_use", message["stop_reason"])

	content, contentIsSlice := message["content"].([]any)
	require.True(t, contentIsSlice)
	require.Len(t, content, 2)
	textBlock, textBlockIsMap := content[0].(map[string]any)
	require.True(t, textBlockIsMap)
	toolBlock, toolBlockIsMap := content[1].(map[string]any)
	require.True(t, toolBlockIsMap)
	usage, usageIsMap := message["usage"].(map[string]any)
	require.True(t, usageIsMap)
	assert.Equal(t, "text", textBlock["type"])
	assert.Equal(t, "tool_use", toolBlock["type"])
	assert.Equal(t, "call_123", toolBlock["id"])
	assert.Equal(t, map[string]any{"path": "README.md"}, toolBlock["input"])
	assert.Equal(t, float64(42), usage["input_tokens"])
	assert.Equal(t, int64(len(transformed)), resp.ContentLength)
}

func TestOpenAIProviderTransformsStreamingResponse(t *testing.T) {
	t.Parallel()

	provider := providers.NewOpenAIProvider(testOpenAIConfig())
	source := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"resp_123",` +
			`"model":"gpt-5.6-sol","status":"in_progress",` +
			`"usage":{"input_tokens":12,"output_tokens":0}}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}`,
		``,
		`event: response.content_part.added`,
		`data: {"type":"response.content_part.added","output_index":0,"part":{"type":"output_text","text":""}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"Hello"}`,
		``,
		`event: response.output_item.done`,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":1,` +
			`"item":{"type":"function_call","call_id":"call_1",` +
			`"name":"read_file","arguments":""}}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"path\":\"README.md\"}"}`,
		``,
		`event: response.output_item.done`,
		`data: {"type":"response.output_item.done","output_index":1,` +
			`"item":{"type":"function_call","call_id":"call_1",` +
			`"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"resp_123",` +
			`"model":"gpt-5.6-sol","status":"completed",` +
			`"usage":{"input_tokens":12,"output_tokens":9}}}`,
		``,
	}, "\n")
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{providers.ContentTypeSSE}},
		Body:       io.NopCloser(strings.NewReader(source)),
	}

	require.NoError(t, provider.TransformHTTPResponse(resp))
	transformed, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	stream := string(transformed)

	assert.Contains(t, stream, "event: message_start")
	assert.Contains(t, stream, `"text":"Hello"`)
	assert.Contains(t, stream, `"type":"text_delta"`)
	assert.Contains(t, stream, `"id":"call_1"`)
	assert.Contains(t, stream, `"name":"read_file"`)
	assert.Contains(t, stream, `"type":"tool_use"`)
	assert.Contains(t, stream, `"type":"input_json_delta"`)
	assert.Contains(t, stream, `"stop_reason":"tool_use"`)
	assert.Contains(t, stream, `"output_tokens":9`)
	assert.Contains(t, stream, "event: message_stop")
}

func TestOpenAIProviderTransformsErrorResponse(t *testing.T) {
	t.Parallel()

	provider := providers.NewOpenAIProvider(testOpenAIConfig())
	source := `{"error":{"message":"Rate limit reached","type":"rate_limit_error"}}`
	resp := &http.Response{
		StatusCode: http.StatusTooManyRequests,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(source)),
	}

	require.NoError(t, provider.TransformHTTPResponse(resp))
	transformed, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"type":"error",
		"error":{"type":"rate_limit_error","message":"Rate limit reached"}
	}`, string(transformed))
}
