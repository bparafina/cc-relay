// Package providers provides shared transformation utilities for cloud providers.
package providers

import (
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ExtractModel extracts the model field from a JSON request body.
// Returns empty string if model field is not present.
func ExtractModel(body []byte) string {
	return gjson.GetBytes(body, "model").String()
}

// RemoveModelFromBody removes the model field from a JSON request body.
// Used by Bedrock/Vertex which put model in URL path, not body.
func RemoveModelFromBody(body []byte) ([]byte, error) {
	return sjson.DeleteBytes(body, "model")
}

// AddAnthropicVersion adds or updates the anthropic_version field in the request body.
// Bedrock uses "bedrock-2023-05-31", Vertex uses "vertex-2023-10-16".
func AddAnthropicVersion(body []byte, version string) ([]byte, error) {
	return sjson.SetBytes(body, "anthropic_version", version)
}

// IsStreamingRequest checks if the request body has "stream": true.
// Returns false if stream field is missing or not a boolean.
func IsStreamingRequest(body []byte) bool {
	result := gjson.GetBytes(body, "stream")
	return result.Exists() && result.Bool()
}

// TransformBodyForCloudProvider performs the standard transformation for cloud providers:
// 1. Extract model (for URL construction)
// 2. Remove model from body
// 3. Add anthropic_version to body
// Returns the modified body and the extracted model name.
func TransformBodyForCloudProvider(
	body []byte,
	anthropicVersion string,
) (newBody []byte, model string, err error) {
	model = ExtractModel(body)

	newBody, err = RemoveModelFromBody(body)
	if err != nil {
		return nil, "", err
	}

	newBody, err = AddAnthropicVersion(newBody, anthropicVersion)
	if err != nil {
		return nil, "", err
	}

	return newBody, model, nil
}

// bedrockUnsupportedFields are top-level Anthropic Messages API fields that
// Bedrock's InvokeModel schema rejects with "Extra inputs are not permitted".
// Clients talking to the relay think it's the real Anthropic API and send
// them; strip before forwarding. "stream" is expressed by the
// invoke-with-response-stream endpoint instead of the body.
var bedrockUnsupportedFields = []string{
	"stream",
	"context_management",
	"betas",
}

// SanitizeBodyForBedrock removes request fields Bedrock's InvokeModel schema
// rejects and translates thinking config to the form Bedrock accepts.
// Bedrock-specific: Vertex/Azure accept the standard Anthropic schema.
func SanitizeBodyForBedrock(body []byte) ([]byte, error) {
	var err error
	for _, field := range bedrockUnsupportedFields {
		body, err = sjson.DeleteBytes(body, field)
		if err != nil {
			return nil, err
		}
	}

	// Bedrock rejects {"type":"enabled"} thinking on models that only
	// support adaptive thinking. Translate the Anthropic form.
	if gjson.GetBytes(body, "thinking.type").String() == "enabled" {
		body, err = sjson.SetBytes(body, "thinking", map[string]string{"type": "adaptive"})
		if err != nil {
			return nil, err
		}
	}

	return body, nil
}
