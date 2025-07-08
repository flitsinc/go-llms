package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/flitsinc/go-llms/tools"
)

// MCPTool implements the tools.Tool interface for MCP-based tools
type MCPTool struct {
	client  *Client
	mcpTool Tool
	schema  *tools.FunctionSchema
	label   string
}

// NewMCPTool creates a new MCPTool from an MCP Tool and client
func NewMCPTool(client *Client, mcpTool Tool) *MCPTool {
	// Convert MCP tool schema to tools.FunctionSchema
	schema := convertMCPSchemaToFunctionSchema(mcpTool)

	// Create a human-readable label
	label := mcpTool.Name
	if mcpTool.Description != "" {
		label = mcpTool.Description
	}

	return &MCPTool{
		client:  client,
		mcpTool: mcpTool,
		schema:  schema,
		label:   label,
	}
}

// Label returns a human-readable title for the tool
func (t *MCPTool) Label() string {
	return t.label
}

// Description returns the description of the tool
func (t *MCPTool) Description() string {
	return t.mcpTool.Description
}

// FuncName returns the function name for the tool
func (t *MCPTool) FuncName() string {
	return t.mcpTool.Name
}

// Run executes the tool with the provided parameters
func (t *MCPTool) Run(r tools.Runner, params json.RawMessage) tools.Result {
	// Convert json.RawMessage to map[string]interface{} for MCP call
	var args map[string]interface{}
	if len(params) > 0 {
		if err := json.Unmarshal(params, &args); err != nil {
			return tools.ErrorWithLabel("Parameter parsing failed",
				fmt.Errorf("failed to parse parameters for %s: %w", t.mcpTool.Name, err))
		}
	}

	// Call the MCP tool
	resp, err := t.client.CallTool(r.Context(), t.mcpTool.Name, args)
	if err != nil {
		return tools.ErrorWithLabel("MCP tool execution failed",
			fmt.Errorf("failed to call MCP tool %s: %w", t.mcpTool.Name, err))
	}

	// Handle error response from MCP server
	if resp.IsError {
		errorMsg := "MCP tool returned error"
		if len(resp.Content) > 0 {
			errorMsg = resp.Content[0].Text
		}
		return tools.ErrorWithLabel("MCP tool error", fmt.Errorf("%s", errorMsg))
	}

	// Convert MCP response to tools.Result
	if len(resp.Content) == 0 {
		return tools.Success(map[string]interface{}{"result": "success"})
	}

	// If single text content, return it directly
	if len(resp.Content) == 1 && resp.Content[0].Type == "text" {
		return tools.SuccessWithLabel(t.mcpTool.Name, map[string]interface{}{
			"result": resp.Content[0].Text,
		})
	}

	// Multiple contents, return as structured data
	contentData := make([]map[string]interface{}, len(resp.Content))
	for i, content := range resp.Content {
		contentData[i] = map[string]interface{}{
			"type": content.Type,
			"text": content.Text,
		}
	}

	return tools.SuccessWithLabel(t.mcpTool.Name, map[string]interface{}{
		"content": contentData,
	})
}

// Schema returns the JSON schema for the tool
func (t *MCPTool) Schema() *tools.FunctionSchema {
	return t.schema
}

// convertMCPSchemaToFunctionSchema converts an MCP tool schema to tools.FunctionSchema
func convertMCPSchemaToFunctionSchema(mcpTool Tool) *tools.FunctionSchema {
	schema := &tools.FunctionSchema{
		Name:        mcpTool.Name,
		Description: mcpTool.Description,
		Parameters:  convertMCPInputSchemaToValueSchema(mcpTool.InputSchema),
	}

	return schema
}

// convertMCPInputSchemaToValueSchema converts MCP input schema to tools.ValueSchema
func convertMCPInputSchemaToValueSchema(inputSchema map[string]interface{}) tools.ValueSchema {
	// Default to object type if not specified
	schemaType := "object"
	if t, ok := inputSchema["type"].(string); ok {
		schemaType = t
	}

	schema := tools.ValueSchema{
		Type: schemaType,
	}

	// Handle description
	if desc, ok := inputSchema["description"].(string); ok {
		schema.Description = desc
	}

	// Handle properties for object type
	if schemaType == "object" {
		if props, ok := inputSchema["properties"].(map[string]interface{}); ok {
			properties := make(map[string]tools.ValueSchema)
			for name, propSchema := range props {
				if propMap, ok := propSchema.(map[string]interface{}); ok {
					properties[name] = convertMCPInputSchemaToValueSchema(propMap)
				}
			}
			schema.Properties = &properties
		}

		// Handle required fields
		if req, ok := inputSchema["required"].([]interface{}); ok {
			required := make([]string, 0, len(req))
			for _, r := range req {
				if s, ok := r.(string); ok {
					required = append(required, s)
				}
			}
			schema.Required = required
		}

		// Handle additionalProperties
		if ap, ok := inputSchema["additionalProperties"]; ok {
			schema.AdditionalProperties = ap
		}
	}

	// Handle array items
	if schemaType == "array" {
		if items, ok := inputSchema["items"].(map[string]interface{}); ok {
			itemSchema := convertMCPInputSchemaToValueSchema(items)
			schema.Items = &itemSchema
		}
	}

	return schema
}

// ConnectServer connects to an MCP server and returns all available tools
func ConnectServer(ctx context.Context, config TransportConfig) ([]tools.Tool, error) {
	client, err := ConnectClient(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MCP server: %w", err)
	}
	mcpTools, err := client.ListTools(ctx)
	if err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to list MCP tools: %w", err)
	}
	// Convert MCP tools to tools.Tool interface
	toolsList := make([]tools.Tool, 0, len(mcpTools))
	for _, mcpTool := range mcpTools {
		tool := NewMCPTool(client, mcpTool)
		toolsList = append(toolsList, tool)
	}
	return toolsList, nil
}

// ConnectStdioServer is a convenience function to connect to a stdio-based MCP server
func ConnectStdioServer(ctx context.Context, command string, args ...string) ([]tools.Tool, error) {
	config := TransportConfig{
		Type:    "stdio",
		Command: command,
		Args:    args,
	}
	return ConnectServer(ctx, config)
}

// ConnectTCPServer is a convenience function to connect to a TCP-based MCP server
func ConnectTCPServer(ctx context.Context, host string, port int) ([]tools.Tool, error) {
	config := TransportConfig{
		Type: "tcp",
		Host: host,
		Port: port,
	}
	return ConnectServer(ctx, config)
}

// MCPServerManager manages connections to multiple MCP servers
type MCPServerManager struct {
	clients map[string]*Client
	tools   map[string][]tools.Tool
}

// NewMCPServerManager creates a new server manager
func NewMCPServerManager() *MCPServerManager {
	return &MCPServerManager{
		clients: make(map[string]*Client),
		tools:   make(map[string][]tools.Tool),
	}
}

// AddServer adds an MCP server and returns its tools
func (m *MCPServerManager) AddServer(ctx context.Context, name string, config TransportConfig) ([]tools.Tool, error) {
	if _, exists := m.clients[name]; exists {
		return nil, fmt.Errorf("server %s already exists", name)
	}

	client, err := ConnectClient(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MCP server %s: %w", name, err)
	}

	mcpTools, err := client.ListTools(ctx)
	if err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to list tools for server %s: %w", name, err)
	}

	// Convert MCP tools to tools.Tool interface
	toolsList := make([]tools.Tool, 0, len(mcpTools))
	for _, mcpTool := range mcpTools {
		tool := NewMCPTool(client, mcpTool)
		toolsList = append(toolsList, tool)
	}

	m.clients[name] = client
	m.tools[name] = toolsList

	return toolsList, nil
}

// RemoveServer removes an MCP server and closes its connection
func (m *MCPServerManager) RemoveServer(name string) error {
	client, exists := m.clients[name]
	if !exists {
		return fmt.Errorf("server %s not found", name)
	}

	if err := client.Close(); err != nil {
		return fmt.Errorf("failed to close connection to server %s: %w", name, err)
	}

	delete(m.clients, name)
	delete(m.tools, name)

	return nil
}

// GetAllTools returns all tools from all connected servers
func (m *MCPServerManager) GetAllTools() []tools.Tool {
	var allTools []tools.Tool
	for _, toolsList := range m.tools {
		allTools = append(allTools, toolsList...)
	}
	return allTools
}

// GetServerTools returns tools from a specific server
func (m *MCPServerManager) GetServerTools(name string) ([]tools.Tool, error) {
	tools, exists := m.tools[name]
	if !exists {
		return nil, fmt.Errorf("server %s not found", name)
	}
	return tools, nil
}

// ListServers returns names of all connected servers
func (m *MCPServerManager) ListServers() []string {
	names := make([]string, 0, len(m.clients))
	for name := range m.clients {
		names = append(names, name)
	}
	return names
}

// Close closes all server connections
func (m *MCPServerManager) Close() error {
	var errs []string
	for name, client := range m.clients {
		if err := client.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("failed to close %s: %v", name, err))
		}
	}

	m.clients = make(map[string]*Client)
	m.tools = make(map[string][]tools.Tool)

	if len(errs) > 0 {
		return fmt.Errorf("errors closing servers: %s", strings.Join(errs, ", "))
	}

	return nil
}
