# Atomic Contracts

**A conversation-to-contract prototype built for LLMxLaw at Stanford (2025).**

Atomic Contracts explores a simple question: *what if an agreement could be made clearer before it became a contract?*

Many everyday agreements begin in informal conversations—between collaborators, clients, lenders, roommates, or small businesses. The important terms may be present, but scattered: a price is mentioned in one message, a deadline in another, and assumptions about responsibility or risk may never be stated explicitly. Turning that exchange into a contract often happens late, after ambiguity has already taken root.

This project investigates how an LLM can help bridge that gap. Rather than treating contract drafting as a one-step text-generation task, it uses the conversation to identify where the parties may lack clarity, confidence, or alignment, then uses that input to shape a more appropriate first draft.

## The idea

The core idea is to create **atomic contracts**: agreements assembled from the concrete commitments in a conversation, with the level of protection calibrated to the parties’ own assessment of trust and certainty.

The prototype follows three stages:

1. **Read the conversation.** A user pastes an informal conversation or transcript describing a potential agreement.
2. **Surface ambiguity and assess trust.** The model generates a small set of neutral questions about unclear or potentially contested points. The user rates each point from 1 (very low confidence/clarity/agreement) to 5 (very high).
3. **Draft a preliminary agreement.** Based on the original conversation and the resulting High, Medium, or Low trust assessment, the model produces a contract draft that extracts the discussed parties, money, dates, deliverables, and other terms.

The trust level is not intended as a judgment about a person. It is a practical signal for how much specificity, verification, and risk allocation the draft may need.

| Trust assessment | Intended drafting approach |
| --- | --- |
| High | A collaborative, balanced draft centered on the agreed core terms. |
| Medium | Clearer definitions, milestones, payment checkpoints, and transparency mechanisms. |
| Low | More explicit breach terms, verification, safeguards, and formal dispute-resolution options. |

## What we were trying to pursue

Atomic Contracts was an experiment in **agreement formation**, not just document generation. The intended value was to help people:

- turn an unstructured discussion into an explicit set of commitments;
- identify unanswered questions before they become disputes;
- preserve the details actually discussed instead of replacing them with generic boilerplate;
- choose contractual safeguards that reflect uncertainty in the specific relationship; and
- leave with a usable starting point for discussion and professional legal review.

In short, the project treats a contract as a living record of mutual understanding—one that should help parties discover and resolve misalignment before signing.

## Prototype flow

```text
Informal conversation
        ↓
LLM-generated clarification / trust questions
        ↓
Participant ratings (1–5)
        ↓
High / Medium / Low trust assessment
        ↓
Preliminary contract draft tailored to the conversation and assessment
```

## Current implementation

The current proof of concept is a [Streamlit](https://streamlit.io/) application powered by Google's Gemini model. It:

- accepts a pasted conversation transcript;
- generates 4–5 questions to probe clarity, certainty, fairness, and agreement;
- collects scaled responses;
- categorizes the overall assessment as High, Medium, or Low; and
- drafts a structured preliminary contract outline using details extracted from the conversation.

## Running locally

Install the dependencies and provide a Gemini API key in a local `.env` file:

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env
streamlit run app.py
```

## Important limitations

This was a hackathon prototype, not a legal-service product. Generated drafts are starting points for discussion and should be reviewed by qualified counsel before being relied on or signed. The prototype's trust assessment is also a lightweight interaction design experiment—not a legal, financial, or behavioral-risk determination.

## Future direction

The broader direction was to make agreement-building more iterative and transparent: support multiple parties, show the source conversation behind each proposed clause, let parties negotiate individual terms, track changes and consent, and bring legal review into the loop where it matters most.

