package providers

import (
	"bytes"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
)

const (
	// DefaultOpenAIBaseURL is the default base URL for the OpenAI API.
	DefaultOpenAIBaseURL = "https://api.openai.com/v1"

	// OpenAIOwner is the owner identifier used in model-list responses.
	OpenAIOwner = "openai"

	// DefaultOpenAIReasoningEffort is used when no provider-specific effort is configured.
	DefaultOpenAIReasoningEffort = "medium"

	jsonFieldType  = "type"
	jsonFieldName  = "name"
	jsonFieldText  = "text"
	jsonFieldIndex = "index"
	jsonFieldDelta = "delta"

	roleUser      = "user"
	roleAssistant = "assistant"

	itemTypeMessage            = "message"
	itemTypeFunctionCall       = "function_call"
	itemTypeFunctionCallOutput = "function_call_output"
	itemTypeOutputText         = "output_text"
	itemTypeReasoning          = "reasoning"

	blockTypeText       = "text"
	blockTypeToolUse    = "tool_use"
	blockTypeToolResult = "tool_result"
	blockTypeThinking   = "thinking"
	blockTypeRedacted   = "redacted_thinking"

	eventTypeError             = "error"
	eventTypeContentBlockDelta = "content_block_delta"
)

// DefaultOpenAIModels are advertised when no explicit model list is configured.
var DefaultOpenAIModels = []string{"gpt-5.6-sol"}

// OpenAIConfig configures an OpenAI Responses API provider.
type OpenAIConfig struct {
	ModelMapping    map[string]string
	Name            string
	BaseURL         string
	ReasoningEffort string
	Models          []string
}

// OpenAIProvider translates between the Anthropic Messages API exposed by
// cc-relay and OpenAI's Responses API.
type OpenAIProvider struct {
	reasoningEffort string
	BaseProvider
}

// NewOpenAIProvider creates an OpenAI Responses API provider.
func NewOpenAIProvider(cfg *OpenAIConfig) *OpenAIProvider {
	if cfg == nil {
		cfg = &OpenAIConfig{
			ModelMapping:    nil,
			Name:            "",
			BaseURL:         "",
			ReasoningEffort: "",
			Models:          nil,
		}
	}

	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultOpenAIBaseURL
	}

	models := cfg.Models
	if len(models) == 0 {
		models = DefaultOpenAIModels
	}

	reasoningEffort := cfg.ReasoningEffort
	if reasoningEffort == "" {
		reasoningEffort = DefaultOpenAIReasoningEffort
	}

	return &OpenAIProvider{
		BaseProvider: NewBaseProviderWithMapping(
			cfg.Name,
			baseURL,
			OpenAIOwner,
			models,
			cfg.ModelMapping,
		),
		reasoningEffort: reasoningEffort,
	}
}

// Authenticate adds OpenAI bearer-token authentication.
func (p *OpenAIProvider) Authenticate(req *http.Request, key string) error {
	req.Header.Set("Authorization", "Bearer "+key)
	return nil
}

// ForwardHeaders returns the headers accepted by the OpenAI API.
func (p *OpenAIProvider) ForwardHeaders(_ http.Header) http.Header {
	headers := make(http.Header)
	headers.Set("Content-Type", "application/json")
	headers.Set("Accept", "application/json")
	return headers
}

// RequiresBodyTransform reports that Anthropic Messages requests must be
// translated to the Responses API wire format.
func (p *OpenAIProvider) RequiresBodyTransform() bool {
	return true
}

// TransformRequest converts an Anthropic Messages request to an OpenAI
// Responses request and redirects it to /v1/responses.
func (p *OpenAIProvider) TransformRequest(
	body []byte,
	_ string,
) (newBody []byte, targetURL string, err error) {
	newBody, err = anthropicToOpenAIRequest(body, p.reasoningEffort, p.MapModel)
	if err != nil {
		return nil, "", fmt.Errorf("openai: transform request: %w", err)
	}
	return newBody, p.baseURL + "/responses", nil
}

// TransformHTTPResponse converts OpenAI JSON or SSE responses to Anthropic
// Messages wire format.
func (p *OpenAIProvider) TransformHTTPResponse(resp *http.Response) error {
	if resp == nil || resp.Body == nil {
		return nil
	}

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return transformOpenAIErrorResponse(resp)
	}

	mediaType, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if err == nil && mediaType == ContentTypeSSE {
		resp.Body = newOpenAIToAnthropicSSEBody(resp.Body)
		resp.Header.Del("Content-Length")
		resp.ContentLength = -1
		return nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response body: %w", err)
	}
	if closeErr := resp.Body.Close(); closeErr != nil {
		return fmt.Errorf("close response body: %w", closeErr)
	}

	transformed, err := openAIToAnthropicResponse(body)
	if err != nil {
		return err
	}
	resp.Body = io.NopCloser(bytes.NewReader(transformed))
	resp.ContentLength = int64(len(transformed))
	resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(transformed)))
	resp.Header.Set("Content-Type", "application/json")
	return nil
}

func transformOpenAIErrorResponse(resp *http.Response) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read OpenAI error response: %w", err)
	}
	if closeErr := resp.Body.Close(); closeErr != nil {
		return fmt.Errorf("close OpenAI error response: %w", closeErr)
	}

	transformed := openAIErrorToAnthropic(body, resp.StatusCode)
	resp.Body = io.NopCloser(bytes.NewReader(transformed))
	resp.ContentLength = int64(len(transformed))
	resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(transformed)))
	resp.Header.Set("Content-Type", "application/json")
	return nil
}
