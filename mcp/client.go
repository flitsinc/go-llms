package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Client represents an MCP client connection to a server
type Client struct {
	transport Transport
	nextID    int64
	pending   map[string]chan JSONRPCResponse
	mu        sync.RWMutex

	// Connection state
	initialized bool
	serverInfo  ServerInfo
	tools       []Tool
	toolsCache  map[string]Tool

	// Configuration
	clientInfo ClientInfo
	timeout    time.Duration
}

// NewClient creates a new MCP client with the given transport
func NewClient(transport Transport) *Client {
	return &Client{
		transport:  transport,
		pending:    make(map[string]chan JSONRPCResponse),
		toolsCache: make(map[string]Tool),
		clientInfo: ClientInfo{
			Name:    "go-llms",
			Version: "1.0.0",
		},
		timeout: 30 * time.Second,
	}
}

// Initialize performs the MCP initialization handshake
func (c *Client) Initialize(ctx context.Context) error {
	if c.initialized {
		return nil
	}

	req := InitializeRequest{
		ProtocolVersion: "2024-11-05",
		Capabilities: map[string]interface{}{
			"tools": map[string]interface{}{},
		},
		ClientInfo: c.clientInfo,
	}

	var resp InitializeResponse
	if err := c.call(ctx, "initialize", req, &resp); err != nil {
		return fmt.Errorf("failed to initialize MCP connection: %w", err)
	}

	c.serverInfo = resp.ServerInfo

	// Send initialized notification
	notif := JSONRPCNotification{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	}

	if err := c.transport.SendNotification(notif); err != nil {
		return fmt.Errorf("failed to send initialized notification: %w", err)
	}

	c.initialized = true

	return nil
}

// ListTools retrieves all available tools from the MCP server
func (c *Client) ListTools(ctx context.Context) ([]Tool, error) {
	if !c.initialized {
		if err := c.Initialize(ctx); err != nil {
			return nil, err
		}
	}

	req := ListToolsRequest{}
	var resp ListToolsResponse

	if err := c.call(ctx, "tools/list", req, &resp); err != nil {
		return nil, fmt.Errorf("failed to list tools: %w", err)
	}

	c.tools = resp.Tools

	// Update cache
	c.mu.Lock()
	c.toolsCache = make(map[string]Tool)
	for _, tool := range c.tools {
		c.toolsCache[tool.Name] = tool
	}
	c.mu.Unlock()

	return c.tools, nil
}

// CallTool executes a tool with the given arguments
func (c *Client) CallTool(ctx context.Context, name string, args map[string]interface{}) (*CallToolResponse, error) {
	if !c.initialized {
		if err := c.Initialize(ctx); err != nil {
			return nil, err
		}
	}

	req := CallToolRequest{
		Name:      name,
		Arguments: args,
	}

	var resp CallToolResponse
	if err := c.call(ctx, "tools/call", req, &resp); err != nil {
		return nil, fmt.Errorf("failed to call tool %s: %w", name, err)
	}

	return &resp, nil
}

// GetTool returns a cached tool by name
func (c *Client) GetTool(name string) (Tool, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	tool, exists := c.toolsCache[name]
	return tool, exists
}

// Close closes the MCP client connection
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Close all pending channels
	for _, ch := range c.pending {
		close(ch)
	}
	c.pending = make(map[string]chan JSONRPCResponse)

	return c.transport.Close()
}

// call performs a JSON-RPC method call
func (c *Client) call(ctx context.Context, method string, params interface{}, result interface{}) error {
	idNum := atomic.AddInt64(&c.nextID, 1)
	id := NewStringID(fmt.Sprintf("%d", idNum))

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	// Create response channel
	respChan := make(chan JSONRPCResponse, 1)

	c.mu.Lock()
	c.pending[id.String()] = respChan
	c.mu.Unlock()

	// Cleanup on exit
	defer func() {
		c.mu.Lock()
		delete(c.pending, id.String())
		c.mu.Unlock()
		close(respChan)
	}()

	// Send request
	if err := c.transport.Send(req); err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}

	// Wait for response with timeout
	select {
	case resp := <-respChan:
		if resp.Error != nil {
			return resp.Error
		}

		// Unmarshal result if provided
		if result != nil && resp.Result != nil {
			resultBytes, err := json.Marshal(resp.Result)
			if err != nil {
				return fmt.Errorf("failed to marshal result: %w", err)
			}

			if err := json.Unmarshal(resultBytes, result); err != nil {
				return fmt.Errorf("failed to unmarshal result: %w", err)
			}
		}

		return nil

	case <-ctx.Done():
		return ctx.Err()

	case <-time.After(c.timeout):
		return fmt.Errorf("request timeout after %v", c.timeout)
	}
}

// startResponseHandler starts a goroutine to handle incoming responses
func (c *Client) startResponseHandler() {
	go func() {
		for {
			resp, err := c.transport.Receive()
			if err != nil {
				// Connection closed or error occurred
				return
			}

			c.mu.RLock()
			respChan, exists := c.pending[resp.ID.String()]
			c.mu.RUnlock()

			if exists {
				select {
				case respChan <- resp:
				default:
					// Channel full or closed
				}
			}
		}
	}()
}

// SetTimeout sets the request timeout
func (c *Client) SetTimeout(timeout time.Duration) {
	c.timeout = timeout
}

// ServerInfo returns information about the connected MCP server
func (c *Client) ServerInfo() ServerInfo {
	return c.serverInfo
}
