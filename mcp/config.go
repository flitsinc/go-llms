package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

// LoadConfig loads MCP server configurations from a JSON file and returns all tools.
// The JSON format supports both stdio and HTTP-based MCP servers.
func LoadConfig(ctx context.Context, configPath string) (map[string][]tools.Tool, error) {
	file, err := os.Open(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	return LoadConfigFromJSON(ctx, data)
}

// LoadConfigFromJSON loads MCP server configurations from JSON data and returns all tools.
func LoadConfigFromJSON(ctx context.Context, data []byte) (map[string][]tools.Tool, error) {
	var config MCPConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config JSON: %w", err)
	}

	result := make(map[string][]tools.Tool)

	// Use errgroup to connect to all servers concurrently.
	var mu sync.Mutex
	g, ctx := errgroup.WithContext(ctx)

	for serverName, serverConfig := range config.MCPServers {
		g.Go(func() error {
			transportConfig, err := convertToTransportConfig(serverConfig)
			if err != nil {
				return fmt.Errorf("invalid config for server %s: %w", serverName, err)
			}
			toolsList, err := ConnectServer(ctx, transportConfig)
			if err != nil {
				return fmt.Errorf("failed to connect to server %s: %w", serverName, err)
			}
			mu.Lock()
			result[serverName] = toolsList
			mu.Unlock()
			return nil
		})
	}

	// Wait for all connections to complete.
	if err := g.Wait(); err != nil {
		return nil, err
	}

	return result, nil
}

// LoadConfigToLLM loads MCP servers from a config file and adds all tools to the LLM.
func LoadConfigToLLM(ctx context.Context, llm *llms.LLM, configPath string) error {
	serverTools, err := LoadConfig(ctx, configPath)
	if err != nil {
		return err
	}

	for _, toolsList := range serverTools {
		for _, tool := range toolsList {
			llm.AddTool(tool)
		}
	}

	return nil
}

// convertToTransportConfig converts an MCPServerConfig to a TransportConfig
func convertToTransportConfig(config MCPServerConfig) (TransportConfig, error) {
	var transportConfig TransportConfig

	// Determine transport type based on available fields
	if config.Command != "" {
		// Stdio transport
		transportConfig.Type = "stdio"
		transportConfig.Command = config.Command
		transportConfig.Args = config.Args
		transportConfig.Env = config.Env
	} else if config.URL != "" {
		// HTTP transport
		transportConfig.Type = "http"
		transportConfig.URL = config.URL
		transportConfig.Headers = config.Headers
	} else if config.Host != "" && config.Port > 0 {
		// TCP transport
		transportConfig.Type = "tcp"
		transportConfig.Host = config.Host
		transportConfig.Port = config.Port
	} else {
		return transportConfig, fmt.Errorf("server config must specify either command, url, or host+port")
	}

	return transportConfig, nil
}

// GenerateExampleConfig creates an example MCP configuration file
func GenerateExampleConfig(path string) error {
	config := MCPConfig{
		MCPServers: map[string]MCPServerConfig{
			"filesystem": {
				Command: "npx",
				Args:    []string{"-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"},
				Env: map[string]string{
					"NODE_ENV": "production",
				},
			},
			"web-search": {
				URL: "http://localhost:3000/mcp",
				Headers: map[string]string{
					"Authorization": "Bearer your-api-key",
					"Content-Type":  "application/json",
				},
			},
			"database": {
				Host: "localhost",
				Port: 8080,
			},
		},
	}

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal example config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write example config: %w", err)
	}

	return nil
}
