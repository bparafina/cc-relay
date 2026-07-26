// Package proxy implements the HTTP proxy server for cc-relay.
package proxy

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/rs/zerolog/log"
	"github.com/samber/lo"

	"github.com/omarluq/cc-relay/internal/config"
	"github.com/omarluq/cc-relay/internal/keypool"
	"github.com/omarluq/cc-relay/internal/providers"
)

type requestTransformErrorContextKey struct{}

func getRequestTransformError(req *http.Request) error {
	if req == nil {
		return nil
	}
	transformErr, isError := req.Context().Value(requestTransformErrorContextKey{}).(error)
	if !isError {
		return nil
	}
	return transformErr
}

func markRequestTransformError(req *http.Request, transformErr error) *http.Request {
	return req.WithContext(context.WithValue(
		req.Context(),
		requestTransformErrorContextKey{},
		transformErr,
	))
}

// ModifyResponseFunc is a callback for additional response processing.
type ModifyResponseFunc func(resp *http.Response) error

// ProviderProxy bundles a provider with its dedicated reverse proxy.
// Each proxy has the provider's URL and auth baked in at creation time,
// ensuring requests are routed to the correct backend with correct authentication.
type ProviderProxy struct {
	Provider           providers.Provider
	Proxy              *httputil.ReverseProxy
	KeyPool            *keypool.KeyPool
	targetURL          *url.URL
	modifyResponseHook ModifyResponseFunc
	APIKey             string `json:"-"`
	debugOpts          config.DebugOptions
}

// NewProviderProxy creates a provider-specific proxy with correct URL and auth.
// The proxy is configured to use this provider's BaseURL for all requests.
// The modifyResponseHook is called after SSE header handling for additional processing.
func NewProviderProxy(
	provider providers.Provider,
	apiKey string,
	pool *keypool.KeyPool,
	debugOpts config.DebugOptions,
	modifyResponseHook ModifyResponseFunc,
) (*ProviderProxy, error) {
	targetURL, err := url.Parse(provider.BaseURL())
	if err != nil {
		return nil, fmt.Errorf("invalid provider base URL %q: %w", provider.BaseURL(), err)
	}

	providerProxy := &ProviderProxy{
		Provider:           provider,
		Proxy:              nil,
		KeyPool:            pool,
		APIKey:             apiKey,
		debugOpts:          debugOpts,
		targetURL:          targetURL,
		modifyResponseHook: modifyResponseHook,
	}

	providerProxy.Proxy = &httputil.ReverseProxy{
		Rewrite:        providerProxy.rewrite,
		FlushInterval:  -1, // Immediate flush for SSE
		ModifyResponse: providerProxy.modifyResponse,
		ErrorHandler: func(w http.ResponseWriter, req *http.Request, _ error) {
			if transformErr := getRequestTransformError(req); transformErr != nil {
				WriteError(w, http.StatusBadRequest, "invalid_request_error", transformErr.Error())
				return
			}
			WriteError(w, http.StatusBadGateway, "api_error", "upstream connection failed")
		},
	}

	return providerProxy, nil
}

// modifyResponse handles SSE headers, Event Stream conversion, and calls the optional hook.
func (pp *ProviderProxy) modifyResponse(resp *http.Response) error {
	if err := pp.transformProviderHTTPResponse(resp); err != nil {
		return err
	}

	pp.normalizeStreamingResponse(resp)

	// Call the hook for additional processing (key pool updates, circuit breaker)
	if pp.modifyResponseHook != nil {
		return pp.modifyResponseHook(resp)
	}

	return nil
}

func (pp *ProviderProxy) normalizeStreamingResponse(resp *http.Response) {
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		return
	}
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return
	}
	if mediaType == providers.ContentTypeSSE {
		SetSSEHeaders(resp.Header)
	}
	if pp.Provider.StreamingContentType() != providers.ContentTypeEventStream ||
		mediaType != providers.ContentTypeEventStream {
		return
	}

	resp.Header.Set("Content-Type", providers.ContentTypeSSE)
	SetSSEHeaders(resp.Header)
	if resp.Body != nil {
		resp.Body = newEventStreamToSSEBody(resp.Body)
	}
}

func (pp *ProviderProxy) transformProviderHTTPResponse(resp *http.Response) error {
	transformer, needsTransform := pp.Provider.(providers.HTTPResponseTransformer)
	if !needsTransform {
		return nil
	}
	if err := transformer.TransformHTTPResponse(resp); err != nil {
		return fmt.Errorf("transform provider response: %w", err)
	}
	return nil
}

// rewrite creates the Rewrite function for this provider's proxy.
func (pp *ProviderProxy) rewrite(proxyRequest *httputil.ProxyRequest) {
	// Handle body transformation for cloud providers (Bedrock, Vertex)
	// This must happen before SetURL because cloud providers return a dynamic target URL
	if pp.Provider.RequiresBodyTransform() {
		pp.rewriteWithTransform(proxyRequest)
		return
	}

	// Standard providers: use static target URL
	proxyRequest.SetURL(pp.targetURL)
	proxyRequest.SetXForwarded()
	pp.setAuth(proxyRequest)
}

// rewriteWithTransform handles cloud providers that need body transformation.
// Cloud providers like Bedrock and Vertex need to:
// 1. Extract model from request body
// 2. Remove model from body and add anthropic_version
// 3. Construct dynamic URL with model in path.
func (pp *ProviderProxy) rewriteWithTransform(proxyRequest *httputil.ProxyRequest) {
	// Read the original body
	var originalBody []byte
	if proxyRequest.In.Body != nil {
		var err error
		originalBody, err = io.ReadAll(proxyRequest.In.Body)
		if closeErr := proxyRequest.In.Body.Close(); closeErr != nil {
			log.Error().Err(closeErr).Msg("failed to close request body")
		}
		if err != nil {
			pp.failRequestTransform(proxyRequest, fmt.Errorf("read request body: %w", err))
			return
		}
	}

	// Get the endpoint path for transform (e.g., "/v1/messages")
	endpoint := proxyRequest.In.URL.Path

	// Transform the request body and get the dynamic target URL
	newBody, targetURLStr, err := pp.Provider.TransformRequest(originalBody, endpoint)
	if err != nil {
		pp.failRequestTransform(proxyRequest, err)
		return
	}

	// Parse the dynamic target URL returned by the provider
	targetURL, err := url.Parse(targetURLStr)
	if err != nil {
		pp.failRequestTransform(proxyRequest, fmt.Errorf("invalid transformed target URL: %w", err))
		return
	}

	// Set the transformed body
	proxyRequest.Out.Body = io.NopCloser(bytes.NewReader(newBody))
	proxyRequest.Out.ContentLength = int64(len(newBody))

	// For cloud providers, the targetURL contains the complete path including model.
	// We set proxyRequest.Out.URL directly instead of using SetURL which would append the original path.
	proxyRequest.Out.URL = targetURL
	proxyRequest.Out.Host = targetURL.Host
	proxyRequest.SetXForwarded()
	pp.setAuth(proxyRequest)
}

// failRequestTransform prevents an incompatible request body from being sent
// upstream. ReverseProxy's Rewrite callback cannot return an error, so route
// the request to a guaranteed-invalid local target and attach the real error
// for ErrorHandler to return in Anthropic format.
func (pp *ProviderProxy) failRequestTransform(proxyRequest *httputil.ProxyRequest, transformErr error) {
	proxyRequest.Out = markRequestTransformError(proxyRequest.Out, transformErr)
	proxyRequest.Out.URL = &url.URL{
		Scheme: "http",
		Host:   "127.0.0.1:0",
		Path:   "/",
	}
	proxyRequest.Out.Host = "127.0.0.1:0"
	proxyRequest.Out.Body = http.NoBody
	proxyRequest.Out.ContentLength = 0
}

// setAuth handles authentication and header forwarding.
func (pp *ProviderProxy) setAuth(proxyReq *httputil.ProxyRequest) {
	// Remove internal header before proxying to avoid key leakage
	proxyReq.Out.Header.Del("X-Selected-Key")

	clientAuth := proxyReq.In.Header.Get("Authorization")
	clientAPIKey := proxyReq.In.Header.Get("x-api-key")
	hasClientAuth := clientAuth != "" || clientAPIKey != ""

	if hasClientAuth && pp.Provider.SupportsTransparentAuth() {
		// TRANSPARENT MODE: Client has auth AND provider accepts it
		// Forward client auth unchanged alongside anthropic-* headers
		lo.ForEach(lo.Entries(proxyReq.In.Header), func(entry lo.Entry[string, []string], _ int) {
			canonicalKey := http.CanonicalHeaderKey(entry.Key)
			if len(canonicalKey) >= 10 && canonicalKey[:10] == "Anthropic-" {
				proxyReq.Out.Header[canonicalKey] = entry.Value
			}
		})
		proxyReq.Out.Header.Set("Content-Type", "application/json")
	} else {
		// CONFIGURED KEY MODE: Use our configured keys
		// Either client has no auth, or provider doesn't accept client auth
		proxyReq.Out.Header.Del("Authorization")
		proxyReq.Out.Header.Del("x-api-key")

		// Get the selected API key from context (set in ServeHTTP via header)
		selectedKey := proxyReq.In.Header.Get("X-Selected-Key")
		if selectedKey == "" {
			selectedKey = pp.APIKey // Fallback to single-key mode
		}

		// Only authenticate if we have a key to use
		if selectedKey != "" {
			if err := pp.Provider.Authenticate(proxyReq.Out, selectedKey); err != nil {
				log.Error().
					Err(err).
					Str("provider", pp.Provider.Name()).
					Msg("failed to authenticate request")
			}
		}
		// If no key available, let backend return 401 (transparent error)

		// Forward anthropic-* headers
		forwardHeaders := pp.Provider.ForwardHeaders(proxyReq.In.Header)
		lo.ForEach(lo.Entries(forwardHeaders), func(entry lo.Entry[string, []string], _ int) {
			proxyReq.Out.Header[entry.Key] = entry.Value
		})
	}
}

// eventStreamToSSEBody wraps an Event Stream body and converts it to SSE on read.
// This allows the ReverseProxy to transparently convert Bedrock Event Stream
// to SSE format without requiring a custom transport.
type eventStreamToSSEBody struct {
	original  io.ReadCloser
	sseBuffer *bytes.Buffer // Buffered SSE output
	esBuffer  []byte        // Accumulated Event Stream data for parsing
	done      bool
}

// newEventStreamToSSEBody creates a wrapper that converts Event Stream to SSE.
func newEventStreamToSSEBody(original io.ReadCloser) *eventStreamToSSEBody {
	return &eventStreamToSSEBody{
		original:  original,
		sseBuffer: bytes.NewBuffer(nil),
		esBuffer:  make([]byte, 0, 32*1024),
		done:      false,
	}
}

// Read implements io.Reader, converting Event Stream messages to SSE events.
func (e *eventStreamToSSEBody) Read(p []byte) (int, error) {
	for {
		if e.sseBuffer.Len() > 0 {
			return e.sseBuffer.Read(p)
		}
		if e.done {
			return 0, io.EOF
		}

		readErr := e.readAndBuffer()
		if errors.Is(readErr, io.EOF) {
			if e.sseBuffer.Len() == 0 {
				return 0, io.EOF
			}
			continue
		}
		if readErr != nil {
			return 0, readErr
		}
	}
}

func (e *eventStreamToSSEBody) readAndBuffer() error {
	chunk := make([]byte, 16*1024) // 16KB chunks
	n, readErr := e.original.Read(chunk)
	if n > 0 {
		e.esBuffer = append(e.esBuffer, chunk[:n]...)
	}

	if n == 0 && readErr == nil {
		return ErrStreamClosed
	}

	e.parseEventStreamBuffer()

	if readErr == io.EOF {
		e.done = true
		return io.EOF
	}
	return readErr
}

func (e *eventStreamToSSEBody) parseEventStreamBuffer() {
	for {
		msg, consumed, err := providers.ParseEventStreamMessage(e.esBuffer)
		if err != nil {
			return
		}
		e.esBuffer = e.esBuffer[consumed:]
		e.sseBuffer.Write(providers.FormatMessageAsSSE(msg))
	}
}

// Close implements io.Closer.
func (e *eventStreamToSSEBody) Close() error {
	return e.original.Close()
}
