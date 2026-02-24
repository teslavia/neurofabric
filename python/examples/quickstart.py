#!/usr/bin/env python3
"""NeuralOS quickstart — demonstrates Generator API."""

from neuralos import Generator


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: quickstart.py <model.gguf> [prompt]")
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, world!"

    gen = Generator(model_path)
    print(f"Generating with: {prompt}")
    output = gen.generate(prompt, max_tokens=64)
    print(output)

    # Chat example
    messages = [{"role": "user", "content": prompt}]
    response = gen.chat(messages, max_tokens=64)
    print(f"Chat response: {response}")


if __name__ == "__main__":
    main()
