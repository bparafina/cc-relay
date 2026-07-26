package providers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

const maxOpenAIStreamEventBytes = 16 * 1024 * 1024

type openAIToAnthropicSSEBody struct {
	original       io.ReadCloser
	scanner        *bufio.Scanner
	blocks         map[int]*openAIStreamBlock
	responseID     string
	model          string
	buffer         bytes.Buffer
	inputTokens    int
	outputTokens   int
	nextBlockIndex int
	messageStarted bool
	hasToolUse     bool
	done           bool
}

type openAIStreamBlock struct {
	Type             string
	CallID           string
	Name             string
	AnthropicIndex   int
	SawArgumentDelta bool
}

type openAIStreamEvent struct {
	Part        openAIResponseContent `json:"part"`
	Error       openAIStreamError     `json:"error"`
	Type        string                `json:"type"`
	Delta       string                `json:"delta"`
	Item        openAIResponseOutput  `json:"item"`
	Response    openAIResponse        `json:"response"`
	OutputIndex int                   `json:"output_index"`
}

type openAIStreamError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

func newOpenAIToAnthropicSSEBody(original io.ReadCloser) *openAIToAnthropicSSEBody {
	scanner := bufio.NewScanner(original)
	scanner.Buffer(make([]byte, 64*1024), maxOpenAIStreamEventBytes)
	return &openAIToAnthropicSSEBody{
		original:       original,
		scanner:        scanner,
		blocks:         make(map[int]*openAIStreamBlock),
		responseID:     "",
		model:          "",
		buffer:         bytes.Buffer{},
		inputTokens:    0,
		outputTokens:   0,
		nextBlockIndex: 0,
		messageStarted: false,
		hasToolUse:     false,
		done:           false,
	}
}

func (body *openAIToAnthropicSSEBody) Read(target []byte) (int, error) {
	for body.buffer.Len() == 0 && !body.done {
		if err := body.readAndConvertEvent(); err != nil {
			return 0, err
		}
	}
	if body.buffer.Len() > 0 {
		return body.buffer.Read(target)
	}
	return 0, io.EOF
}

func (body *openAIToAnthropicSSEBody) Close() error {
	body.done = true
	return body.original.Close()
}

func (body *openAIToAnthropicSSEBody) readAndConvertEvent() error {
	eventType, data, found, err := body.readSSEEvent()
	if err != nil {
		return err
	}
	if !found {
		body.done = true
		return nil
	}
	if len(data) == 0 || string(data) == "[DONE]" {
		return nil
	}

	var event openAIStreamEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return fmt.Errorf("openai: decode stream event: %w", err)
	}
	if event.Type == "" {
		event.Type = eventType
	}
	return body.convertEvent(&event)
}

func (body *openAIToAnthropicSSEBody) readSSEEvent() (
	eventType string,
	data []byte,
	found bool,
	err error,
) {
	var dataLines []string
	for body.scanner.Scan() {
		line := body.scanner.Text()
		if line == "" {
			if eventType != "" || len(dataLines) > 0 {
				return eventType, []byte(strings.Join(dataLines, "\n")), true, nil
			}
			continue
		}
		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if scanErr := body.scanner.Err(); scanErr != nil {
		return "", nil, false, fmt.Errorf("openai: read stream: %w", scanErr)
	}
	if eventType != "" || len(dataLines) > 0 {
		return eventType, []byte(strings.Join(dataLines, "\n")), true, nil
	}
	return "", nil, false, nil
}

func (body *openAIToAnthropicSSEBody) convertEvent(event *openAIStreamEvent) error {
	switch event.Type {
	case "response.created", "response.in_progress":
		body.captureResponse(&event.Response)
		return body.ensureMessageStart()
	case "response.output_item.added":
		return body.addOutputItem(event)
	case "response.content_part.added":
		return body.addContentPart(event)
	case "response.output_text.delta", "response.refusal.delta":
		return body.emitTextDelta(event)
	case "response.function_call_arguments.delta":
		return body.emitFunctionArgumentsDelta(event)
	case "response.output_item.done":
		return body.finishOutputItem(event)
	default:
		return body.convertTerminalEvent(event)
	}
}

func (body *openAIToAnthropicSSEBody) convertTerminalEvent(event *openAIStreamEvent) error {
	switch event.Type {
	case "response.completed":
		body.captureResponse(&event.Response)
		return body.finishResponse("end_turn")
	case "response.incomplete":
		body.captureResponse(&event.Response)
		return body.finishResponse("max_tokens")
	case "response.failed":
		return body.emitFailedResponse(event)
	case eventTypeError:
		message := event.Error.Message
		if message == "" {
			message = "OpenAI stream failed"
		}
		return body.emitError(message)
	default:
		return nil
	}
}

func (body *openAIToAnthropicSSEBody) addOutputItem(event *openAIStreamEvent) error {
	if event.Item.Type != itemTypeFunctionCall {
		return nil
	}
	body.hasToolUse = true
	return body.startBlock(event.OutputIndex, blockTypeToolUse, event.Item.CallID, event.Item.Name)
}

func (body *openAIToAnthropicSSEBody) addContentPart(event *openAIStreamEvent) error {
	if event.Part.Type != itemTypeOutputText && event.Part.Type != "refusal" {
		return nil
	}
	return body.startBlock(event.OutputIndex, blockTypeText, "", "")
}

func (body *openAIToAnthropicSSEBody) emitTextDelta(event *openAIStreamEvent) error {
	if err := body.startBlock(event.OutputIndex, blockTypeText, "", ""); err != nil {
		return err
	}
	return body.emit(eventTypeContentBlockDelta, map[string]any{
		jsonFieldType:  eventTypeContentBlockDelta,
		jsonFieldIndex: body.blocks[event.OutputIndex].AnthropicIndex,
		jsonFieldDelta: map[string]any{
			jsonFieldType: "text_delta",
			jsonFieldText: event.Delta,
		},
	})
}

func (body *openAIToAnthropicSSEBody) emitFunctionArgumentsDelta(event *openAIStreamEvent) error {
	block, blockExists := body.blocks[event.OutputIndex]
	if !blockExists {
		return fmt.Errorf("openai: function argument delta arrived before function call item")
	}
	block.SawArgumentDelta = true
	return body.emit(eventTypeContentBlockDelta, functionArgumentsDelta(block, event.Delta))
}

func functionArgumentsDelta(block *openAIStreamBlock, arguments string) map[string]any {
	return map[string]any{
		jsonFieldType:  eventTypeContentBlockDelta,
		jsonFieldIndex: block.AnthropicIndex,
		jsonFieldDelta: map[string]any{
			jsonFieldType:  "input_json_delta",
			"partial_json": arguments,
		},
	}
}

func (body *openAIToAnthropicSSEBody) finishOutputItem(event *openAIStreamEvent) error {
	block, blockExists := body.blocks[event.OutputIndex]
	if blockExists &&
		block.Type == blockTypeToolUse &&
		!block.SawArgumentDelta &&
		event.Item.Arguments != "" {
		if err := body.emit(
			eventTypeContentBlockDelta,
			functionArgumentsDelta(block, event.Item.Arguments),
		); err != nil {
			return err
		}
	}
	return body.stopBlock(event.OutputIndex)
}

func (body *openAIToAnthropicSSEBody) emitFailedResponse(event *openAIStreamEvent) error {
	message := "OpenAI response failed"
	if event.Response.IncompleteDetails != nil &&
		event.Response.IncompleteDetails.Reason != "" {
		message = event.Response.IncompleteDetails.Reason
	}
	return body.emitError(message)
}

func (body *openAIToAnthropicSSEBody) captureResponse(response *openAIResponse) {
	if response == nil {
		return
	}
	if response.ID != "" {
		body.responseID = response.ID
	}
	if response.Model != "" {
		body.model = response.Model
	}
	body.inputTokens = response.Usage.InputTokens
	body.outputTokens = response.Usage.OutputTokens
}

func (body *openAIToAnthropicSSEBody) ensureMessageStart() error {
	if body.messageStarted {
		return nil
	}
	body.messageStarted = true
	return body.emit("message_start", map[string]any{
		jsonFieldType: "message_start",
		itemTypeMessage: map[string]any{
			"id":            body.responseID,
			jsonFieldType:   itemTypeMessage,
			"role":          roleAssistant,
			"model":         body.model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  body.inputTokens,
				"output_tokens": 0,
			},
		},
	})
}

func (body *openAIToAnthropicSSEBody) startBlock(
	outputIndex int,
	blockType string,
	callID string,
	name string,
) error {
	if _, exists := body.blocks[outputIndex]; exists {
		return nil
	}
	if err := body.ensureMessageStart(); err != nil {
		return err
	}

	block := &openAIStreamBlock{
		Type:             blockType,
		CallID:           callID,
		Name:             name,
		AnthropicIndex:   body.nextBlockIndex,
		SawArgumentDelta: false,
	}
	body.nextBlockIndex++
	body.blocks[outputIndex] = block

	contentBlock := map[string]any{
		jsonFieldType: blockType,
		jsonFieldText: "",
	}
	if blockType == blockTypeToolUse {
		contentBlock = map[string]any{
			jsonFieldType: blockTypeToolUse,
			"id":          callID,
			jsonFieldName: name,
			"input":       map[string]any{},
		}
	}

	return body.emit("content_block_start", map[string]any{
		jsonFieldType:   "content_block_start",
		jsonFieldIndex:  block.AnthropicIndex,
		"content_block": contentBlock,
	})
}

func (body *openAIToAnthropicSSEBody) stopBlock(outputIndex int) error {
	block, ok := body.blocks[outputIndex]
	if !ok {
		return nil
	}
	if err := body.emit("content_block_stop", map[string]any{
		jsonFieldType:  "content_block_stop",
		jsonFieldIndex: block.AnthropicIndex,
	}); err != nil {
		return err
	}
	delete(body.blocks, outputIndex)
	return nil
}

func (body *openAIToAnthropicSSEBody) finishResponse(defaultStopReason string) error {
	if err := body.ensureMessageStart(); err != nil {
		return err
	}

	outputIndexes := make([]int, 0, len(body.blocks))
	for outputIndex := range body.blocks {
		outputIndexes = append(outputIndexes, outputIndex)
	}
	sort.Slice(outputIndexes, func(i, j int) bool {
		return body.blocks[outputIndexes[i]].AnthropicIndex <
			body.blocks[outputIndexes[j]].AnthropicIndex
	})
	for _, outputIndex := range outputIndexes {
		if err := body.stopBlock(outputIndex); err != nil {
			return err
		}
	}

	stopReason := defaultStopReason
	if body.hasToolUse {
		stopReason = "tool_use"
	}
	if err := body.emit("message_delta", map[string]any{
		jsonFieldType: "message_delta",
		jsonFieldDelta: map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"output_tokens": body.outputTokens,
		},
	}); err != nil {
		return err
	}
	if err := body.emit("message_stop", map[string]any{jsonFieldType: "message_stop"}); err != nil {
		return err
	}
	body.done = true
	return nil
}

func (body *openAIToAnthropicSSEBody) emitError(message string) error {
	if err := body.emit(eventTypeError, map[string]any{
		jsonFieldType: eventTypeError,
		eventTypeError: map[string]any{
			jsonFieldType: "api_error",
			"message":     message,
		},
	}); err != nil {
		return err
	}
	body.done = true
	return nil
}

func (body *openAIToAnthropicSSEBody) emit(eventType string, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("openai: encode Anthropic stream event: %w", err)
	}
	_, _ = fmt.Fprintf(&body.buffer, "event: %s\ndata: %s\n\n", eventType, encoded)
	return nil
}
