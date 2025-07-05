package mcp

import (
	"context"
	"testing"
)

func TestLoadConfigFromJSON(t *testing.T) {
	ctx := context.Background()
	
	// Test stdio server config
	stdioJSON := `{
		"mcpServers": {
			"filesystem": {
				"command": "python",
				"args": ["-m", "mcp_server"],
				"env": {
					"API_KEY": "test-key",
					"DEBUG": "true"
				}
			}
		}
	}`
	
	_, err := LoadConfigFromJSON(ctx, []byte(stdioJSON))
	// We expect this to fail since the server doesn't exist, but config parsing should work
	if err == nil {
		t.Error("Expected error due to non-existent server, but got nil")
	}
	if err != nil && err.Error() == "failed to parse config JSON" {
		t.Errorf("Config parsing failed: %v", err)
	}
	
	// Test HTTP server config
	httpJSON := `{
		"mcpServers": {
			"web-api": {
				"url": "http://localhost:3000/mcp",
				"headers": {
					"Authorization": "Bearer token",
					"Content-Type": "application/json"
				}
			}
		}
	}`
	
	_, err = LoadConfigFromJSON(ctx, []byte(httpJSON))
	// We expect this to fail since the server doesn't exist, but config parsing should work
	if err == nil {
		t.Error("Expected error due to non-existent server, but got nil")
	}
	if err != nil && err.Error() == "failed to parse config JSON" {
		t.Errorf("Config parsing failed: %v", err)
	}
	
	// Test TCP server config
	tcpJSON := `{
		"mcpServers": {
			"database": {
				"host": "localhost",
				"port": 8080
			}
		}
	}`
	
	_, err = LoadConfigFromJSON(ctx, []byte(tcpJSON))
	// We expect this to fail since the server doesn't exist, but config parsing should work
	if err == nil {
		t.Error("Expected error due to non-existent server, but got nil")
	}
	if err != nil && err.Error() == "failed to parse config JSON" {
		t.Errorf("Config parsing failed: %v", err)
	}
}

func TestConvertToTransportConfig(t *testing.T) {
	// Test stdio config conversion
	stdioConfig := MCPServerConfig{
		Command: "python",
		Args:    []string{"-m", "server"},
		Env:     map[string]string{"KEY": "value"},
	}
	
	transport, err := convertToTransportConfig(stdioConfig)
	if err != nil {
		t.Errorf("Failed to convert stdio config: %v", err)
	}
	
	if transport.Type != "stdio" {
		t.Errorf("Expected type 'stdio', got '%s'", transport.Type)
	}
	
	if transport.Command != "python" {
		t.Errorf("Expected command 'python', got '%s'", transport.Command)
	}
	
	// Test HTTP config conversion
	httpConfig := MCPServerConfig{
		URL:     "http://localhost:3000/mcp",
		Headers: map[string]string{"Auth": "Bearer token"},
	}
	
	transport, err = convertToTransportConfig(httpConfig)
	if err != nil {
		t.Errorf("Failed to convert HTTP config: %v", err)
	}
	
	if transport.Type != "http" {
		t.Errorf("Expected type 'http', got '%s'", transport.Type)
	}
	
	if transport.URL != "http://localhost:3000/mcp" {
		t.Errorf("Expected URL 'http://localhost:3000/mcp', got '%s'", transport.URL)
	}
	
	// Test TCP config conversion
	tcpConfig := MCPServerConfig{
		Host: "localhost",
		Port: 8080,
	}
	
	transport, err = convertToTransportConfig(tcpConfig)
	if err != nil {
		t.Errorf("Failed to convert TCP config: %v", err)
	}
	
	if transport.Type != "tcp" {
		t.Errorf("Expected type 'tcp', got '%s'", transport.Type)
	}
	
	if transport.Host != "localhost" {
		t.Errorf("Expected host 'localhost', got '%s'", transport.Host)
	}
	
	if transport.Port != 8080 {
		t.Errorf("Expected port 8080, got %d", transport.Port)
	}
	
	// Test invalid config
	invalidConfig := MCPServerConfig{}
	
	_, err = convertToTransportConfig(invalidConfig)
	if err == nil {
		t.Error("Expected error for invalid config, but got nil")
	}
}

func TestGenerateExampleConfig(t *testing.T) {
	tmpFile := "/tmp/test_mcp_config.json"
	
	err := GenerateExampleConfig(tmpFile)
	if err != nil {
		t.Errorf("Failed to generate example config: %v", err)
	}
	
	// Try to load the generated config
	ctx := context.Background()
	_, err = LoadConfig(ctx, tmpFile)
	// We expect this to fail due to non-existent servers, but parsing should work
	if err != nil && err.Error() == "failed to parse config JSON" {
		t.Errorf("Generated config is invalid: %v", err)
	}
}