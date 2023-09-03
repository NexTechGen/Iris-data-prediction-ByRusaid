import streamlit as st
import pickle
from PIL import Image
from sklearn import datasets
import pandas as pd

lin_model=pickle.load(open('lin_model.pkl','rb'))
log_model=pickle.load(open('log_model.pkl','rb'))
svm=pickle.load(open('svm.pkl','rb'))

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Type"] = iris.target

df.loc[df['Type'] == 0, "target_names"]='Iris-setosa'
df.loc[df['Type'] == 1, "target_names"]='Iris-versicolor'
df.loc[df['Type'] == 2, "target_names"]='Iris-virginica'


def classify(num):
    if num < 0.5:
        return 'Iris-Setosa'
    elif num < 1.5:
        return 'Iris-Versicolor'
    else:
        return 'Iris-Virginica'
def main():
    st.title("Streamlit Tutorial")

    image = Image.open('iris.jpg')
    st.image(image)

    st.markdown(
        """<style>
                div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
                                                                                            font-size: 20px;
        </style>
        """, unsafe_allow_html=True)


    activities=['Linear Regression','Logistic Regression','SVM']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)

    sl=st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length', 0.0, 10.0)
    pw=st.slider('Select Petal Width', 0.0, 10.0)
    st.markdown("##")

    if st.button('Classify'):
        if sl == 0.00 or sw == 0.00 or pl == 0.00 or pw == 0.00:
            st.warning("You Fool Enter parameters")
        else:
            inputs=[[sl,sw,pl,pw]]
            if option=='Linear Regression':
                st.success(classify(lin_model.predict(inputs)))
            elif option=='Logistic Regression':
                st.success(classify(log_model.predict(inputs)))
            else:
                st.success(classify(svm.predict(inputs)))

    tab1, tab2 = st.tabs(["Data", "Owner"])

    with tab1:
        st.dataframe(df.drop('Type', axis=1), hide_index=True)
        st.markdown(f"{df.shape[0]} rows and {df.shape[1]} columns.")
    with tab2:
        html_temp = """
                    <div style="background-color:teal;padding:10px">
                    <h2 style="color:white;text-align:center;">By: <a href="https://nextechgen.github.io" style="color:white; font-style: italic;">Rusaid Ahamed</a></h2>
                    </div>
                    """

        st.markdown(html_temp, unsafe_allow_html=True)


if __name__=='__main__':
    main()
