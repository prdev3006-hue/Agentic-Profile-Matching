# Conversation test scenarios

Run `python matching_agent.py` and enter these scenarios in order, or use each
as a separate CLI session.

1. **Initial JD:** “Hiring for a React engineer, must have React and 3+ years experience; SQL is nice to have.” Check ranked report, evidence, gaps, and stated experience.
2. **Natural language search:** “Find me candidates with React and 3+ years experience.” Check React candidates and the years criterion.
3. **Compare:** “Compare the top 3 matches side by side.” Check the JSON evidence table.
4. **Explain:** “Why did Jane Smith rank higher than Marcus Johnson?” Check the explanation references score evidence, skill coverage, and experience.
5. **Refine mid-conversation:** “Make TypeScript a must-have and require at least 4 years.” Check that the list is re-ranked using the revised criteria.
6. **Screening questions and report export:** “Generate interview questions for Jane Smith” and “Save the report.” Check questions target evidence/gaps and file appears under `outputs/`.

The examples use a local corpus of 100 synthetic resumes (TXT); the file tools
also read PDF and DOCX files.

7. **Second round:** “Run second round deep analysis.” Check the shortlist is at most ten and each profile includes evidence text.
8. **Final round:** “Run final round hire/no-hire recommendation.” Check the report states HIRE or NO-HIRE and its evidence threshold.
9. **RAG evidence retrieval:** Search for React, AWS, and experience; confirm matching resume snippets rank above unrelated files.
