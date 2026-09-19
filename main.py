import streamlit as st
import langchain_helper

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox( "Pick a cuisine", ("Select a cuisine", "Indian", "Mexican", "Italian", "Chinese") )

if cuisine != "Select a cuisine":
    try:
        with st.spinner("Generating restaurant name and menu..."):
            response = langchain_helper.get_restaurant_name_and_items(cuisine)

        st.header(response['restaurant_name'].strip())
        menu_items = response['items'].strip().split(",")
        st.write("**Menu Items**")

        for item in menu_items:
            st.write("-", item)

    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            st.warning("API quota exceeded. Please try again later.")
        else:
            st.error("Something went wrong. Please try again.")
