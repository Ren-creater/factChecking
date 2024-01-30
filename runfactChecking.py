from factChecking import fact_check, acc, process
# import openai

# openai.api_key = " …"

# # first get answer from chatgpt
# model_id = "gpt-3.5-turbo"
# completion1 = openai.ChatCompletion.create(
# model=model_id,
# messages=[
# {"role": "user", 
#  "content": """
# Consider the following claim for fact-checking:
# The King of the USA lives in the White House.
# Also, consider the following arguments in the context of some argumentative explanation for prediction Refuted in verdict to the claim:
# Start your answer with a true or false, and then immediately put the arguments into a numbered list, do not add anything else."""}
# ]
# )
# answer = completion1.choices[0].message.content
# print(answer)

input_text = """
False

1. The United States does not have a king; it is a federal presidential republic with an elected president.
2. The White House is the official residence of the President of the United States, not a king.
3. As of my last knowledge update in January 2022, the President of the United States is not referred to as a king.
"""

result = fact_check(input_text) # answer / input_text
resultACC = acc(*process(result))

print("\nAcc: " + resultACC)