package mcp

import (
	"context"

	"github.com/flitsinc/go-llms/llms"
)

// AddServerToLLM connects to a single MCP server described by the provided
// TransportConfig and adds all exposed tools to the given LLM.
//
// This keeps the dependency pointing from the mcp package → llms, so the llms
// package itself remains completely independent of MCP.
func AddServerToLLM(ctx context.Context, llm *llms.LLM, config TransportConfig) error {
	tools, err := ConnectServer(ctx, config)
	if err != nil {
		return err
	}
	for _, t := range tools {
		llm.AddTool(t)
	}
	return nil
}

// AddStdioServerToLLM is a helper that connects to a stdio-based MCP server
// (e.g. a local subprocess) and adds all its tools to the LLM.
func AddStdioServerToLLM(ctx context.Context, llm *llms.LLM, command string, args ...string) error {
	cfg := TransportConfig{
		Type:    "stdio",
		Command: command,
		Args:    args,
	}
	return AddServerToLLM(ctx, llm, cfg)
}

// AddTCPServerToLLM is a helper that connects to a TCP-based MCP server and
// adds its tools to the LLM.
func AddTCPServerToLLM(ctx context.Context, llm *llms.LLM, host string, port int) error {
	cfg := TransportConfig{
		Type: "tcp",
		Host: host,
		Port: port,
	}
	return AddServerToLLM(ctx, llm, cfg)
}
