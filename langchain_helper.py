from secret_key import GOOGLE_API_KEY
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

import os
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.6)


def get_restaurant_name_and_items(cuisine):

    prompt_name = PromptTemplate.from_template(
        "I want to open a restaurant for {cuisine} food. Suggest one fancy name for this.")
    prompt_items = PromptTemplate.from_template(
        "Suggest me 3 menu items for {restaurant_name}. Give 3 names as comma separated list.")
    parser = StrOutputParser()

    # Step 1: cuisine -> restaurant name
    name_chain = prompt_name | llm | parser

    # Step 2: take that name, feed it as 'restaurant_name' into the next chain
    items_chain = prompt_items | llm | parser

    # Combine: run name_chain - store this in dict with key restaurant_name, then pass its output into items_chain
    full_chain = (
            {"restaurant_name": name_chain}
            | RunnablePassthrough.assign(items=items_chain)
    )

    result = full_chain.invoke({"cuisine": cuisine})

    return result


if __name__ == "__main__":
    print(get_restaurant_name_and_items("Italian"))