/**
 * @file chat_template.hpp
 * @brief Phase 34-D: Chat template parser — ChatML, Llama/Mistral, Phi-3
 *
 * Converts structured messages (system/user/assistant) into model-specific
 * prompt strings. Header-only, no dependencies beyond std.
 */

#ifndef NF_CHAT_TEMPLATE_HPP
#define NF_CHAT_TEMPLATE_HPP

#include <string>
#include <vector>
#include <cstring>

namespace nf {

enum class ChatRole { SYSTEM, USER, ASSISTANT };

struct ChatMessage {
    ChatRole    role;
    std::string content;
};

enum class ChatFormat {
    CHATML,     /* <|im_start|>role\ncontent<|im_end|> */
    LLAMA,      /* [INST] ... [/INST] */
    PHI3,       /* <|user|>\ncontent<|end|> */
    AUTO        /* detect from model architecture */
};

inline const char* chat_role_str(ChatRole r) {
    switch (r) {
        case ChatRole::SYSTEM:    return "system";
        case ChatRole::USER:      return "user";
        case ChatRole::ASSISTANT: return "assistant";
    }
    return "user";
}

/**
 * Detect chat format from architecture name.
 */
inline ChatFormat detect_chat_format(const char* arch) {
    if (!arch) return ChatFormat::CHATML;
    if (std::strcmp(arch, "llama") == 0 || std::strcmp(arch, "mistral") == 0 ||
        std::strcmp(arch, "mixtral") == 0)
        return ChatFormat::LLAMA;
    if (std::strcmp(arch, "phi3") == 0 || std::strcmp(arch, "phi") == 0)
        return ChatFormat::PHI3;
    if (std::strcmp(arch, "qwen2") == 0 || std::strcmp(arch, "gemma") == 0)
        return ChatFormat::CHATML;
    return ChatFormat::CHATML;
}

/**
 * Apply ChatML template.
 * Format: <|im_start|>role\ncontent<|im_end|>\n
 */
inline std::string apply_chatml(const std::vector<ChatMessage>& messages) {
    std::string out;
    for (auto& m : messages) {
        out += "<|im_start|>";
        out += chat_role_str(m.role);
        out += "\n";
        out += m.content;
        out += "<|im_end|>\n";
    }
    /* Add assistant prompt */
    out += "<|im_start|>assistant\n";
    return out;
}

/**
 * Apply Llama/Mistral [INST] template.
 * System message goes in <<SYS>> block inside first [INST].
 * Format: [INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant
 */
inline std::string apply_llama(const std::vector<ChatMessage>& messages) {
    std::string out;
    std::string system_msg;
    bool first_user = true;

    for (auto& m : messages) {
        if (m.role == ChatRole::SYSTEM) {
            system_msg = m.content;
            continue;
        }
        if (m.role == ChatRole::USER) {
            out += "[INST] ";
            if (first_user && !system_msg.empty()) {
                out += "<<SYS>>\n" + system_msg + "\n<</SYS>>\n\n";
            }
            out += m.content;
            out += " [/INST]";
            first_user = false;
        } else if (m.role == ChatRole::ASSISTANT) {
            out += " " + m.content + " ";
        }
    }
    return out;
}

/**
 * Apply Phi-3 template.
 * Format: <|user|>\ncontent<|end|>\n<|assistant|>\n
 */
inline std::string apply_phi3(const std::vector<ChatMessage>& messages) {
    std::string out;
    for (auto& m : messages) {
        if (m.role == ChatRole::SYSTEM) {
            out += "<|system|>\n" + m.content + "<|end|>\n";
        } else if (m.role == ChatRole::USER) {
            out += "<|user|>\n" + m.content + "<|end|>\n";
        } else {
            out += "<|assistant|>\n" + m.content + "<|end|>\n";
        }
    }
    out += "<|assistant|>\n";
    return out;
}

/**
 * Apply chat template to messages.
 * If format is AUTO, detects from arch name.
 */
inline std::string apply_chat_template(
    const std::vector<ChatMessage>& messages,
    ChatFormat format = ChatFormat::AUTO,
    const char* arch = nullptr)
{
    if (format == ChatFormat::AUTO)
        format = detect_chat_format(arch);

    switch (format) {
        case ChatFormat::CHATML: return apply_chatml(messages);
        case ChatFormat::LLAMA:  return apply_llama(messages);
        case ChatFormat::PHI3:   return apply_phi3(messages);
        default:                 return apply_chatml(messages);
    }
}

/**
 * Parse a raw chat string into messages.
 * Supports ChatML format: <|im_start|>role\ncontent<|im_end|>
 */
inline std::vector<ChatMessage> parse_chatml(const std::string& text) {
    std::vector<ChatMessage> msgs;
    const std::string start_tag = "<|im_start|>";
    const std::string end_tag = "<|im_end|>";

    size_t pos = 0;
    while (pos < text.size()) {
        size_t s = text.find(start_tag, pos);
        if (s == std::string::npos) break;
        s += start_tag.size();

        size_t nl = text.find('\n', s);
        if (nl == std::string::npos) break;
        std::string role_str = text.substr(s, nl - s);

        size_t e = text.find(end_tag, nl + 1);
        if (e == std::string::npos) break;
        std::string content = text.substr(nl + 1, e - nl - 1);

        ChatRole role = ChatRole::USER;
        if (role_str == "system") role = ChatRole::SYSTEM;
        else if (role_str == "assistant") role = ChatRole::ASSISTANT;

        msgs.push_back({role, content});
        pos = e + end_tag.size();
    }
    return msgs;
}

} /* namespace nf */

#endif /* NF_CHAT_TEMPLATE_HPP */
