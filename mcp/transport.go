package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os/exec"
	"sync"
	"time"
)

// StdioTransport implements the Transport interface for stdio-based communication
type StdioTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr io.ReadCloser

	encoder *json.Encoder
	decoder *json.Decoder
	mu      sync.Mutex
}

// NewStdioTransport creates a new stdio transport that launches a subprocess
func NewStdioTransport(command string, args ...string) (*StdioTransport, error) {
	return NewStdioTransportWithEnv(command, nil, args...)
}

// NewStdioTransportWithEnv creates a new stdio transport with environment variables
func NewStdioTransportWithEnv(command string, env map[string]string, args ...string) (*StdioTransport, error) {
	cmd := exec.Command(command, args...)

	// Set environment variables
	for key, value := range env {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdin.Close()
		stdout.Close()
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		stderr.Close()
		return nil, fmt.Errorf("failed to start MCP server process: %w", err)
	}

	transport := &StdioTransport{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  stdout,
		stderr:  stderr,
		encoder: json.NewEncoder(stdin),
		decoder: json.NewDecoder(stdout),
	}
	// Note: We don't use decoder.UseNumber() because our custom MCPID type
	// handles both string and numeric IDs correctly with proper marshaling

	return transport, nil
}

func (t *StdioTransport) Send(request JSONRPCRequest) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if err := t.encoder.Encode(request); err != nil {
		return fmt.Errorf("failed to encode request: %w", err)
	}

	return nil
}

func (t *StdioTransport) Receive() (JSONRPCResponse, error) {
	var response JSONRPCResponse
	if err := t.decoder.Decode(&response); err != nil {
		return response, fmt.Errorf("failed to decode response: %w", err)
	}
	return response, nil
}

func (t *StdioTransport) Close() error {
	var errs []error

	if t.stdin != nil {
		if err := t.stdin.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if t.stdout != nil {
		if err := t.stdout.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if t.stderr != nil {
		if err := t.stderr.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if t.cmd != nil && t.cmd.Process != nil {
		if err := t.cmd.Process.Kill(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing transport: %v", errs)
	}

	return nil
}

// TCPTransport implements the Transport interface for TCP-based communication
type TCPTransport struct {
	conn    net.Conn
	encoder *json.Encoder
	decoder *json.Decoder
	mu      sync.Mutex
}

// NewTCPTransport creates a new TCP transport
func NewTCPTransport(host string, port int) (*TCPTransport, error) {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", host, port), 10*time.Second)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MCP server: %w", err)
	}

	t := &TCPTransport{
		conn:    conn,
		encoder: json.NewEncoder(conn),
		decoder: json.NewDecoder(conn),
	}
	// Note: We don't use decoder.UseNumber() because our custom MCPID type
	// handles both string and numeric IDs correctly with proper marshaling
	return t, nil
}

func (t *TCPTransport) Send(request JSONRPCRequest) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if err := t.encoder.Encode(request); err != nil {
		return fmt.Errorf("failed to encode request: %w", err)
	}

	return nil
}

func (t *TCPTransport) Receive() (JSONRPCResponse, error) {
	var response JSONRPCResponse
	if err := t.decoder.Decode(&response); err != nil {
		return response, fmt.Errorf("failed to decode response: %w", err)
	}
	return response, nil
}

func (t *TCPTransport) Close() error {
	if t.conn != nil {
		return t.conn.Close()
	}
	return nil
}

// ConnectTransport creates a transport based on the provided configuration
func ConnectTransport(config TransportConfig) (Transport, error) {
	switch config.Type {
	case "stdio":
		if config.Command == "" {
			return nil, fmt.Errorf("command is required for stdio transport")
		}
		return NewStdioTransportWithEnv(config.Command, config.Env, config.Args...)

	case "tcp":
		if config.Host == "" {
			config.Host = "localhost"
		}
		if config.Port == 0 {
			return nil, fmt.Errorf("port is required for tcp transport")
		}
		return NewTCPTransport(config.Host, config.Port)

	default:
		return nil, fmt.Errorf("unsupported transport type: %s", config.Type)
	}
}

// ConnectClient creates and initializes an MCP client with the given configuration
func ConnectClient(ctx context.Context, config TransportConfig) (*Client, error) {
	transport, err := ConnectTransport(config)
	if err != nil {
		return nil, err
	}

	client := NewClient(transport)
	client.startResponseHandler()

	if err := client.Initialize(ctx); err != nil {
		transport.Close()
		return nil, err
	}

	return client, nil
}
