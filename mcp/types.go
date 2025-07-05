package mcp

import (
	"encoding/json"
	"fmt"
)

// MCPID represents a JSON-RPC ID that can be either a string or number
type MCPID struct {
	isString bool
	strVal   string
	numVal   float64
}

// NewStringID creates an MCPID from a string
func NewStringID(s string) MCPID {
	return MCPID{isString: true, strVal: s}
}

// NewNumberID creates an MCPID from a number
func NewNumberID(n float64) MCPID {
	return MCPID{isString: false, numVal: n}
}

// String returns a string representation for use as a map key
func (id MCPID) String() string {
	if id.isString {
		return id.strVal
	}
	return fmt.Sprintf("%.0f", id.numVal)
}

// MarshalJSON implements json.Marshaler
func (id MCPID) MarshalJSON() ([]byte, error) {
	if id.isString {
		return json.Marshal(id.strVal)
	}
	return json.Marshal(id.numVal)
}

// UnmarshalJSON implements json.Unmarshaler
func (id *MCPID) UnmarshalJSON(data []byte) error {
	// Try string first
	var strVal string
	if err := json.Unmarshal(data, &strVal); err == nil {
		*id = NewStringID(strVal)
		return nil
	}

	// Try number
	var numVal float64
	if err := json.Unmarshal(data, &numVal); err == nil {
		*id = NewNumberID(numVal)
		return nil
	}

	return fmt.Errorf("ID must be string or number")
}

// JSON-RPC 2.0 message types
type JSONRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      MCPID       `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      MCPID       `json:"id"`
	Result  interface{} `json:"result,omitempty"`
	Error   *JSONRPCError `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

func (e *JSONRPCError) Error() string {
	return fmt.Sprintf("JSON-RPC error %d: %s", e.Code, e.Message)
}

// MCP Protocol types
type InitializeRequest struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities"`
	ClientInfo      ClientInfo             `json:"clientInfo"`
}

type InitializeResponse struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities"`
	ServerInfo      ServerInfo             `json:"serverInfo"`
}

type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type ListToolsRequest struct {
	// Empty for now, but MCP spec allows for cursor-based pagination
}

type ListToolsResponse struct {
	Tools []Tool `json:"tools"`
}

type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

type CallToolRequest struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

type CallToolResponse struct {
	Content []Content `json:"content"`
	IsError bool      `json:"isError,omitempty"`
}

type Content struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Transport interface for different connection types
type Transport interface {
	Send(request JSONRPCRequest) error
	Receive() (JSONRPCResponse, error)
	Close() error
}

// TransportConfig holds configuration for connecting to MCP servers
type TransportConfig struct {
	Type string // "stdio", "tcp", "http", "sse"
	
	// For stdio transport
	Command string
	Args    []string
	Env     map[string]string
	
	// For TCP transport
	Host string
	Port int
	
	// For HTTP/SSE transport
	URL     string
	Headers map[string]string
	
	// Authentication
	Auth *AuthConfig
}

type AuthConfig struct {
	Type string // "bearer", "basic", "apikey"
	Token string
	Username string
	Password string
	APIKey string
}

// MCPConfig represents the structure of an MCP configuration file
type MCPConfig struct {
	MCPServers map[string]MCPServerConfig `json:"mcpServers"`
}

// MCPServerConfig represents a single MCP server configuration
type MCPServerConfig struct {
	// For stdio-based servers
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	
	// For HTTP-based servers
	URL     string            `json:"url,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
	
	// For TCP-based servers
	Host string `json:"host,omitempty"`
	Port int    `json:"port,omitempty"`
}