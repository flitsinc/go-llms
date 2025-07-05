package mcp

import (
	"testing"
)

func TestMCPToolConversion(t *testing.T) {
	// Create a mock MCP tool
	mcpTool := Tool{
		Name:        "test_tool",
		Description: "A test tool for MCP",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"message": map[string]interface{}{
					"type":        "string",
					"description": "A message to process",
				},
				"count": map[string]interface{}{
					"type":        "integer",
					"description": "Number of times to repeat",
				},
			},
			"required": []interface{}{"message"},
		},
	}

	// Test schema conversion
	schema := convertMCPSchemaToFunctionSchema(mcpTool)

	// Verify the schema conversion
	if schema.Name != "test_tool" {
		t.Errorf("Expected name 'test_tool', got '%s'", schema.Name)
	}

	if schema.Description != "A test tool for MCP" {
		t.Errorf("Expected description 'A test tool for MCP', got '%s'", schema.Description)
	}

	if schema.Parameters.Type != "object" {
		t.Errorf("Expected parameters type 'object', got '%s'", schema.Parameters.Type)
	}

	if schema.Parameters.Properties == nil {
		t.Error("Expected properties to be set")
		return
	}

	props := *schema.Parameters.Properties

	// Check message property
	messageSchema, exists := props["message"]
	if !exists {
		t.Error("Expected 'message' property to exist")
	} else {
		if messageSchema.Type != "string" {
			t.Errorf("Expected message type 'string', got '%s'", messageSchema.Type)
		}
		if messageSchema.Description != "A message to process" {
			t.Errorf("Expected message description 'A message to process', got '%s'", messageSchema.Description)
		}
	}

	// Check count property
	countSchema, exists := props["count"]
	if !exists {
		t.Error("Expected 'count' property to exist")
	} else {
		if countSchema.Type != "integer" {
			t.Errorf("Expected count type 'integer', got '%s'", countSchema.Type)
		}
	}

	// Check required fields
	if len(schema.Parameters.Required) != 1 || schema.Parameters.Required[0] != "message" {
		t.Errorf("Expected required fields ['message'], got %v", schema.Parameters.Required)
	}
}

func TestTransportConfig(t *testing.T) {
	// Test stdio transport config
	config := TransportConfig{
		Type:    "stdio",
		Command: "python",
		Args:    []string{"-m", "mcp_server"},
	}

	if config.Type != "stdio" {
		t.Errorf("Expected type 'stdio', got '%s'", config.Type)
	}

	if config.Command != "python" {
		t.Errorf("Expected command 'python', got '%s'", config.Command)
	}

	// Test TCP transport config
	tcpConfig := TransportConfig{
		Type: "tcp",
		Host: "localhost",
		Port: 8080,
	}

	if tcpConfig.Type != "tcp" {
		t.Errorf("Expected type 'tcp', got '%s'", tcpConfig.Type)
	}

	if tcpConfig.Host != "localhost" {
		t.Errorf("Expected host 'localhost', got '%s'", tcpConfig.Host)
	}

	if tcpConfig.Port != 8080 {
		t.Errorf("Expected port 8080, got %d", tcpConfig.Port)
	}
}

// MockTransport implements the Transport interface for testing
type MockTransport struct {
	responses     []JSONRPCResponse
	requests      []JSONRPCRequest
	notifications []JSONRPCNotification
	index         int
}

func NewMockTransport(responses []JSONRPCResponse) *MockTransport {
	return &MockTransport{
		responses:     responses,
		requests:      make([]JSONRPCRequest, 0),
		notifications: make([]JSONRPCNotification, 0),
		index:         0,
	}
}

func (m *MockTransport) Send(request JSONRPCRequest) error {
	m.requests = append(m.requests, request)
	return nil
}

func (m *MockTransport) SendNotification(notification JSONRPCNotification) error {
	m.notifications = append(m.notifications, notification)
	return nil
}

func (m *MockTransport) Receive() (JSONRPCResponse, error) {
	if m.index >= len(m.responses) {
		return JSONRPCResponse{}, &JSONRPCError{Code: -1, Message: "no more responses"}
	}
	resp := m.responses[m.index]
	m.index++
	return resp, nil
}

func (m *MockTransport) Close() error {
	return nil
}

func TestMCPClient(t *testing.T) {
	// Create mock responses for initialization
	initResponse := JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      NewNumberID(1),
		Result: InitializeResponse{
			ProtocolVersion: "2024-11-05",
			Capabilities:    map[string]interface{}{},
			ServerInfo: ServerInfo{
				Name:    "test-server",
				Version: "1.0.0",
			},
		},
	}

	mockTransport := NewMockTransport([]JSONRPCResponse{initResponse})
	client := NewClient(mockTransport)

	// Test client creation
	if client == nil {
		t.Error("Expected client to be created")
		return
	}

	if client.clientInfo.Name != "go-llms" {
		t.Errorf("Expected client name 'go-llms', got '%s'", client.clientInfo.Name)
	}

	if client.initialized {
		t.Error("Expected client to not be initialized initially")
	}
}

func TestJSONRPCError(t *testing.T) {
	err := &JSONRPCError{
		Code:    -32600,
		Message: "Invalid Request",
		Data:    "Additional error data",
	}

	expectedMsg := "JSON-RPC error -32600: Invalid Request"
	if err.Error() != expectedMsg {
		t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
	}
}
