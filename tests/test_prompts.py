from pdf_reader.prompts import follow_up_prompt, qa_prompt


def test_qa_prompt_formats_context_question_and_history() -> None:
    messages = qa_prompt.format_messages(
        context="The PDF states that PDF is on page 1.",
        chat_history="User: What is PDF?",
        question="When was it mentioned?",
    )

    assert len(messages) == 2
    assert "only the provided context" in messages[0].content
    assert "The PDF states that PDF is on page 1." in messages[1].content
    assert "User: What is PDF?" in messages[1].content
    assert "When was it mentioned?" in messages[1].content


def test_follow_up_prompt_formats_standalone_question() -> None:
    messages = follow_up_prompt.format_messages(
        chat_history="User: Explain PDF.\nAssistant: A PDF is a document format.",
        question="Who created it?",
    )

    assert len(messages) == 2
    assert "Standalone question:" in messages[1].content
    assert "Who created it?" in messages[1].content
    assert "Explain PDF." in messages[1].content
