package providers

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type openAIResponse struct {
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	ID     string                 `json:"id"`
	Model  string                 `json:"model"`
	Status string                 `json:"status"`
	Output []openAIResponseOutput `json:"output"`
	Usage  openAIResponseUsage    `json:"usage"`
}

type openAIResponseOutput struct {
	Type      string                  `json:"type"`
	CallID    string                  `json:"call_id"`
	Name      string                  `json:"name"`
	Arguments string                  `json:"arguments"`
	Content   []openAIResponseContent `json:"content"`
}

type openAIResponseContent struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	Refusal string `json:"refusal"`
}

type openAIResponseUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicResponse struct {
	StopSequence any                    `json:"stop_sequence"`
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Role         string                 `json:"role"`
	Model        string                 `json:"model"`
	StopReason   string                 `json:"stop_reason"`
	Content      []any                  `json:"content"`
	Usage        anthropicResponseUsage `json:"usage"`
}

type anthropicResponseUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func openAIToAnthropicResponse(body []byte) ([]byte, error) {
	var source openAIResponse
	if err := json.Unmarshal(body, &source); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}
	if source.ID == "" {
		return nil, fmt.Errorf("openai: response id is missing")
	}

	content := make([]any, 0, len(source.Output))
	hasToolUse := false
	for outputIndex, output := range source.Output {
		converted, isToolUse, err := convertOpenAIOutput(&output, outputIndex)
		if err != nil {
			return nil, err
		}
		content = append(content, converted...)
		hasToolUse = hasToolUse || isToolUse
	}

	stopReason := "end_turn"
	if hasToolUse {
		stopReason = "tool_use"
	} else if source.Status == "incomplete" &&
		source.IncompleteDetails != nil &&
		source.IncompleteDetails.Reason == "max_output_tokens" {
		stopReason = "max_tokens"
	}

	target := anthropicResponse{
		StopSequence: nil,
		ID:           source.ID,
		Type:         itemTypeMessage,
		Role:         roleAssistant,
		Model:        source.Model,
		StopReason:   stopReason,
		Content:      content,
		Usage: anthropicResponseUsage{
			InputTokens:  source.Usage.InputTokens,
			OutputTokens: source.Usage.OutputTokens,
		},
	}
	return json.Marshal(target)
}

func convertOpenAIOutput(
	output *openAIResponseOutput,
	outputIndex int,
) (content []any, isToolUse bool, err error) {
	switch output.Type {
	case itemTypeMessage:
		return convertOpenAIMessageContent(output.Content), false, nil
	case itemTypeFunctionCall:
		var input any
		if err := json.Unmarshal([]byte(output.Arguments), &input); err != nil {
			return nil, false, fmt.Errorf(
				"openai: output[%d] function arguments are invalid JSON: %w",
				outputIndex,
				err,
			)
		}
		return []any{map[string]any{
			jsonFieldType: blockTypeToolUse,
			"id":          output.CallID,
			jsonFieldName: output.Name,
			"input":       input,
		}}, true, nil
	case itemTypeReasoning:
		// OpenAI reasoning items are intentionally not exposed to clients.
		return nil, false, nil
	default:
		return nil, false, nil
	}
}

func convertOpenAIMessageContent(parts []openAIResponseContent) []any {
	content := make([]any, 0, len(parts))
	for _, part := range parts {
		var text string
		switch part.Type {
		case itemTypeOutputText:
			text = part.Text
		case "refusal":
			text = part.Refusal
		default:
			continue
		}
		content = append(content, map[string]any{
			jsonFieldType: blockTypeText,
			jsonFieldText: text,
		})
	}
	return content
}

func openAIErrorToAnthropic(body []byte, statusCode int) []byte {
	var source struct {
		Error struct {
			Code    any    `json:"code"`
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &source); err != nil {
		source.Error.Message = ""
	}

	message := source.Error.Message
	if message == "" {
		message = http.StatusText(statusCode)
	}

	errorType := "api_error"
	switch statusCode {
	case http.StatusBadRequest, http.StatusNotFound, http.StatusUnprocessableEntity:
		errorType = "invalid_request_error"
	case http.StatusUnauthorized, http.StatusForbidden:
		errorType = "authentication_error"
	case http.StatusTooManyRequests:
		errorType = "rate_limit_error"
	}

	target := map[string]any{
		jsonFieldType: eventTypeError,
		eventTypeError: map[string]any{
			jsonFieldType: errorType,
			"message":     message,
		},
	}
	transformed, err := json.Marshal(target)
	if err != nil {
		return []byte(`{"type":"error","error":{"type":"api_error","message":"OpenAI request failed"}}`)
	}
	return transformed
}
