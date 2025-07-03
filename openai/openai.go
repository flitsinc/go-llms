package openai

func New(accessToken, model string) *ChatCompletionsAPI {
	return NewChatCompletionsAPI(accessToken, model)
}
