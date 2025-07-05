package main

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"os"
)

// JSON-RPC structures matching the mcp package types

type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      interface{}   `json:"id"`
	Result  interface{}   `json:"result,omitempty"`
	Error   *JSONRPCError `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

type InitializeResponse struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities"`
	ServerInfo      ServerInfo             `json:"serverInfo"`
}

type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
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

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	enc := json.NewEncoder(os.Stdout)
	for scanner.Scan() {
		var req JSONRPCRequest
		if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
			// ignore invalid input
			continue
		}
		switch req.Method {
		case "initialize":
			resp := JSONRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result: InitializeResponse{
					ProtocolVersion: "2025-06-18",
					Capabilities:    map[string]interface{}{"tools": map[string]interface{}{}},
					ServerInfo:      ServerInfo{Name: "fetch-server", Version: "0.1"},
				},
			}
			enc.Encode(resp)
		case "tools/list":
			tool := Tool{
				Name:        "fetch",
				Description: "Fetch a URL over HTTP",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"url": map[string]interface{}{
							"type":        "string",
							"description": "HTTP URL to fetch",
						},
					},
					"required": []string{"url"},
				},
			}
			resp := JSONRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result:  ListToolsResponse{Tools: []Tool{tool}},
			}
			enc.Encode(resp)
		case "tools/call":
			var callReq CallToolRequest
			json.Unmarshal(req.Params, &callReq)
			if callReq.Name != "fetch" {
				enc.Encode(JSONRPCResponse{
					JSONRPC: "2.0",
					ID:      req.ID,
					Error:   &JSONRPCError{Code: -32601, Message: "tool not found"},
				})
				continue
			}
			url, ok := callReq.Arguments["url"].(string)
			if !ok {
				enc.Encode(JSONRPCResponse{
					JSONRPC: "2.0",
					ID:      req.ID,
					Error:   &JSONRPCError{Code: -32602, Message: "invalid params"},
				})
				continue
			}
			var text string
			if resp, err := http.Get(url); err == nil {
				b, err2 := io.ReadAll(resp.Body)
				resp.Body.Close()
				if err2 == nil {
					text = string(b)
				} else {
					text = err2.Error()
				}
			} else {
				text = err.Error()
			}
			enc.Encode(JSONRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result:  CallToolResponse{Content: []Content{{Type: "text", Text: text}}},
			})
		default:
			enc.Encode(JSONRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Error:   &JSONRPCError{Code: -32601, Message: "method not found"},
			})
		}
	}
}
