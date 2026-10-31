# Matching agent state machine

```mermaid
flowchart LR
    S([START]) --> P[Parse JD and conversation turn]
    P --> E[Extract must-have and nice-to-have criteria]
    E --> Q[Search resume files / local RAG index]
    Q --> R[Rank candidates with evidence and gaps]
    R --> D{Screening round?}
    D -->|Round 1: top 10| G[Generate match report]
    D -->|Round 2| I[Deep evidence review of top 10]
    I --> G
    D -->|Final round| F[Hire / no-hire recommendation]
    F --> G
    G --> H[Human feedback and refinement]
    H -->|new criteria: next chat turn| P
    H --> X([END for current turn])
```

Each turn ends after the feedback node and the CLI accepts another turn using
the same state. Round 1 ranks the whole corpus and returns at most ten
candidates; round 2 adds evidence text for those ten; final round applies an
explicit threshold to produce a hire/no-hire recommendation. The next user
message can refine criteria and triggers a fresh search and ranking. Scores
support human review and are not a hiring decision by themselves. Retrieval
uses local TF-IDF over resume text.
