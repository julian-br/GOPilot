import ollama

MODEL = "qwen3.5:9b"

print(f"GOPilot — model: {MODEL}  (empty input to quit)\n")

messages = []
while True:
    user_input = input("You: ").strip()
    if not user_input:
        break

    messages.append({"role": "user", "content": user_input})
    response = ollama.chat(model=MODEL, messages=messages)
    reply = response.message.content

    messages.append({"role": "assistant", "content": reply})
    print(f"\nAssistant: {reply}\n")
