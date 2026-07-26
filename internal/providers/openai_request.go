package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

type anthropicMessagesRequest struct {
	ToolChoice json.RawMessage    `json:"tool_choice"`
	Thinking   json.RawMessage    `json:"thinking"`
	System     json.RawMessage    `json:"system"`
	Model      string             `json:"model"`
	Messages   []anthropicMessage `json:"messages"`
	Tools      []anthropicTool    `json:"tools"`
	MaxTokens  int                `json:"max_tokens"`
	Stream     bool               `json:"stream"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	ToolUseID string          `json:"tool_use_id"`
	Input     json.RawMessage `json:"input"`
	Content   json.RawMessage `json:"content"`
	Source    json.RawMessage `json:"source"`
	IsError   bool            `json:"is_error"`
}

type openAIResponsesRequest struct {
	ToolChoice        any            `json:"tool_choice,omitempty"`
	Reasoning         map[string]any `json:"reasoning,omitempty"`
	ParallelToolCalls *bool          `json:"parallel_tool_calls,omitempty"`
	Model             string         `json:"model"`
	Instructions      string         `json:"instructions,omitempty"`
	Input             []any          `json:"input"`
	Tools             []any          `json:"tools,omitempty"`
	MaxOutputTokens   int            `json:"max_output_tokens,omitempty"`
	Stream            bool           `json:"stream"`
	Store             bool           `json:"store"`
}

type openAIInputBuilder struct {
	role           string
	items          []any
	messageContent []any
}

func anthropicToOpenAIRequest(
	body []byte,
	reasoningEffort string,
	mapModel func(string) string,
) ([]byte, error) {
	var source anthropicMessagesRequest
	if err := json.Unmarshal(body, &source); err != nil {
		return nil, fmt.Errorf("decode Anthropic request: %w", err)
	}
	if source.Model == "" {
		return nil, errors.New("model is required")
	}
	if len(source.Messages) == 0 {
		return nil, errors.New("messages is required")
	}

	instructions, err := anthropicSystemToInstructions(source.System)
	if err != nil {
		return nil, err
	}

	input, err := anthropicMessagesToOpenAIInput(source.Messages)
	if err != nil {
		return nil, err
	}

	tools, err := anthropicToolsToOpenAI(source.Tools)
	if err != nil {
		return nil, err
	}

	toolChoice, parallelToolCalls, err := anthropicToolChoiceToOpenAI(source.ToolChoice)
	if err != nil {
		return nil, err
	}

	model := source.Model
	if mapModel != nil {
		model = mapModel(model)
	}

	target := openAIResponsesRequest{
		ToolChoice:        toolChoice,
		Reasoning:         map[string]any{"effort": reasoningEffort},
		Model:             model,
		Instructions:      instructions,
		Input:             input,
		Tools:             tools,
		MaxOutputTokens:   source.MaxTokens,
		Stream:            source.Stream,
		Store:             false,
		ParallelToolCalls: parallelToolCalls,
	}

	return json.Marshal(target)
}

func anthropicSystemToInstructions(raw json.RawMessage) (string, error) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", nil
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}

	blocks, err := decodeAnthropicContent(raw)
	if err != nil {
		return "", fmt.Errorf("decode system prompt: %w", err)
	}

	parts := make([]string, 0, len(blocks))
	for blockIndex := range blocks {
		if blocks[blockIndex].Type == blockTypeText && blocks[blockIndex].Text != "" {
			parts = append(parts, blocks[blockIndex].Text)
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

func anthropicMessagesToOpenAIInput(messages []anthropicMessage) ([]any, error) {
	input := make([]any, 0, len(messages))
	for messageIndex, message := range messages {
		if message.Role != roleUser && message.Role != roleAssistant {
			return nil, fmt.Errorf("messages[%d]: unsupported role %q", messageIndex, message.Role)
		}

		blocks, err := decodeAnthropicContent(message.Content)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", messageIndex, err)
		}

		items, err := anthropicMessageToOpenAIItems(message.Role, blocks)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", messageIndex, err)
		}
		input = append(input, items...)
	}
	return input, nil
}

func decodeAnthropicContent(raw json.RawMessage) ([]anthropicContentBlock, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []anthropicContentBlock{{
			Input:     nil,
			Content:   nil,
			Source:    nil,
			Type:      blockTypeText,
			Text:      text,
			ID:        "",
			Name:      "",
			ToolUseID: "",
			IsError:   false,
		}}, nil
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, fmt.Errorf("decode content blocks: %w", err)
	}
	return blocks, nil
}

func anthropicMessageToOpenAIItems(role string, blocks []anthropicContentBlock) ([]any, error) {
	builder := &openAIInputBuilder{
		items:          make([]any, 0, len(blocks)),
		messageContent: make([]any, 0, len(blocks)),
		role:           role,
	}

	for blockIndex := range blocks {
		if err := builder.appendBlock(&blocks[blockIndex], blockIndex); err != nil {
			return nil, err
		}
	}

	builder.flushMessage()
	return builder.items, nil
}

func (builder *openAIInputBuilder) flushMessage() {
	if len(builder.messageContent) == 0 {
		return
	}
	builder.items = append(builder.items, map[string]any{
		jsonFieldType: itemTypeMessage,
		"role":        builder.role,
		"content":     builder.messageContent,
	})
	builder.messageContent = make([]any, 0)
}

func (builder *openAIInputBuilder) appendBlock(block *anthropicContentBlock, blockIndex int) error {
	switch block.Type {
	case blockTypeText:
		builder.appendText(block.Text)
	case "image":
		return builder.appendImage(block.Source, blockIndex)
	case blockTypeToolUse:
		return builder.appendToolUse(block, blockIndex)
	case blockTypeToolResult:
		return builder.appendToolResult(block, blockIndex)
	case blockTypeThinking, blockTypeRedacted:
		// Reasoning from another provider is intentionally not replayed.
	default:
		return fmt.Errorf("content[%d]: unsupported block type %q", blockIndex, block.Type)
	}
	return nil
}

func (builder *openAIInputBuilder) appendText(text string) {
	contentType := "input_text"
	if builder.role == roleAssistant {
		contentType = itemTypeOutputText
	}
	builder.messageContent = append(builder.messageContent, map[string]any{
		jsonFieldType: contentType,
		jsonFieldText: text,
	})
}

func (builder *openAIInputBuilder) appendImage(source json.RawMessage, blockIndex int) error {
	if builder.role != roleUser {
		return fmt.Errorf("content[%d]: assistant image blocks are unsupported", blockIndex)
	}
	imageURL, err := anthropicImageSourceToURL(source)
	if err != nil {
		return fmt.Errorf("content[%d]: %w", blockIndex, err)
	}
	builder.messageContent = append(builder.messageContent, map[string]any{
		jsonFieldType: "input_image",
		"image_url":   imageURL,
	})
	return nil
}

func (builder *openAIInputBuilder) appendToolUse(block *anthropicContentBlock, blockIndex int) error {
	if builder.role != roleAssistant {
		return fmt.Errorf("content[%d]: tool_use requires assistant role", blockIndex)
	}
	builder.flushMessage()
	arguments := "{}"
	if len(bytes.TrimSpace(block.Input)) > 0 {
		arguments = string(block.Input)
	}
	builder.items = append(builder.items, map[string]any{
		jsonFieldType: itemTypeFunctionCall,
		"call_id":     block.ID,
		jsonFieldName: block.Name,
		"arguments":   arguments,
	})
	return nil
}

func (builder *openAIInputBuilder) appendToolResult(block *anthropicContentBlock, blockIndex int) error {
	if builder.role != roleUser {
		return fmt.Errorf("content[%d]: tool_result requires user role", blockIndex)
	}
	builder.flushMessage()
	output, err := anthropicToolResultOutput(block.Content, block.IsError)
	if err != nil {
		return fmt.Errorf("content[%d]: %w", blockIndex, err)
	}
	builder.items = append(builder.items, map[string]any{
		jsonFieldType: itemTypeFunctionCallOutput,
		"call_id":     block.ToolUseID,
		"output":      output,
	})
	return nil
}

func anthropicImageSourceToURL(raw json.RawMessage) (string, error) {
	var source struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
		URL       string `json:"url"`
	}
	if err := json.Unmarshal(raw, &source); err != nil {
		return "", fmt.Errorf("decode image source: %w", err)
	}
	switch source.Type {
	case "base64":
		if source.MediaType == "" || source.Data == "" {
			return "", errors.New("base64 image source requires media_type and data")
		}
		return "data:" + source.MediaType + ";base64," + source.Data, nil
	case "url":
		if source.URL == "" {
			return "", errors.New("URL image source requires url")
		}
		return source.URL, nil
	default:
		return "", fmt.Errorf("unsupported image source type %q", source.Type)
	}
}

func anthropicToolResultOutput(raw json.RawMessage, isError bool) (string, error) {
	var output string
	if err := json.Unmarshal(raw, &output); err == nil {
		return prefixToolResultError(output, isError), nil
	}

	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return prefixToolResultError("", isError), nil
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", fmt.Errorf("decode tool result: %w", err)
	}

	parts := make([]string, 0, len(blocks))
	for blockIndex := range blocks {
		block := &blocks[blockIndex]
		if block.Type == blockTypeText {
			parts = append(parts, block.Text)
			continue
		}
		encoded, err := json.Marshal(block)
		if err != nil {
			return "", fmt.Errorf("encode tool result block: %w", err)
		}
		parts = append(parts, string(encoded))
	}
	output = strings.Join(parts, "\n")
	return prefixToolResultError(output, isError), nil
}

func prefixToolResultError(output string, isError bool) string {
	if !isError {
		return output
	}
	if output == "" {
		return "Error"
	}
	return "Error: " + output
}

func anthropicToolsToOpenAI(tools []anthropicTool) ([]any, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	result := make([]any, 0, len(tools))
	for index, tool := range tools {
		if tool.Name == "" {
			return nil, fmt.Errorf("tools[%d]: name is required", index)
		}
		var parameters any
		if err := json.Unmarshal(tool.InputSchema, &parameters); err != nil {
			return nil, fmt.Errorf("tools[%d]: decode input_schema: %w", index, err)
		}
		result = append(result, map[string]any{
			jsonFieldType: "function",
			jsonFieldName: tool.Name,
			"description": tool.Description,
			"parameters":  parameters,
			"strict":      false,
		})
	}
	return result, nil
}

func anthropicToolChoiceToOpenAI(raw json.RawMessage) (choice any, parallel *bool, err error) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil, nil
	}

	var source struct {
		Type                   string `json:"type"`
		Name                   string `json:"name"`
		DisableParallelToolUse bool   `json:"disable_parallel_tool_use"`
	}
	if err := json.Unmarshal(raw, &source); err != nil {
		return nil, nil, fmt.Errorf("decode tool_choice: %w", err)
	}

	if source.DisableParallelToolUse {
		value := false
		parallel = &value
	}

	return openAIToolChoice(source.Type, source.Name, parallel)
}

func openAIToolChoice(
	choiceType string,
	name string,
	parallel *bool,
) (choice any, parallelChoice *bool, err error) {
	switch choiceType {
	case "", "auto":
		return "auto", parallel, nil
	case "any":
		return "required", parallel, nil
	case "none":
		return "none", parallel, nil
	case "tool":
		if name == "" {
			return nil, nil, errors.New("tool_choice.name is required for type tool")
		}
		return map[string]any{jsonFieldType: "function", jsonFieldName: name}, parallel, nil
	default:
		return nil, nil, fmt.Errorf("unsupported tool_choice type %q", choiceType)
	}
}
