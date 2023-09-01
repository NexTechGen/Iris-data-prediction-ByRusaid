import streamlit as st
from sklearn import datasets
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC


iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Type"] = iris.target

df.loc[df['Type'] == 0, "target_names"]='Iris-setosa'
df.loc[df['Type'] == 1, "target_names"]='Iris-versicolor'
df.loc[df['Type'] == 2, "target_names"]='Iris-virginica'

df.drop('Type', axis=1)

X=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(X,y)
lin_reg=LinearRegression()
log_reg=LogisticRegression()
svc_model=SVC()
lin_reg=lin_reg.fit(x_train,y_train)
log_reg=log_reg.fit(x_train,y_train)
svc_model=svc_model.fit(x_train,y_train)


def classify(num):
    if num < 0.5:
        return 'Iris-Setosa'
    elif num < 1.5:
        return 'Iris-Versicolor'
    else:
        return 'Iris-Virginica'

def main():
    st.title("Iris Classification")

    image = Image.open('iris.jpg')
    st.image(image)

    st.markdown(
        """<style>
                div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
                                                                                            font-size: 20px;
        </style>
        """, unsafe_allow_html=True)


    activities = ['Linear Regression', 'Logistic Regression', 'SVM']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)

    sl = st.slider('Select Sepal Length', 0.0, 10.0)
    sw = st.slider('Select Sepal Width', 0.0, 10.0)
    pl = st.slider('Select Petal Length', 0.0, 10.0)
    pw = st.slider('Select Petal Width', 0.0, 10.0)
    st.markdown("##")

    if st.button('Classify'):
        if sl == 0.00 or sw == 0.00 or pl == 0.00 or pw == 0.00:
            st.warning("You Fool Enter parameters")
        else:
            inputs = [[sl, sw, pl, pw]]
            if option == 'Linear Regression':
                st.success(classify(lin_reg.predict(inputs)))
            elif option == 'Logistic Regression':
                st.success(classify(log_reg.predict(inputs)))
            else:
                st.success(classify(svc_model.predict(inputs)))


    tab1, tab2= st.tabs(["Data","Owner"])

    with tab1:
        st.dataframe(df)
        st.markdown(f"{df.shape[0]} rows and {df.shape[1]} columns." )
    with tab2:
        html_temp = """
                <div style="background-color:teal;padding:10px">
                <h2 style="color:white;text-align:center;">By: <a href="https://nextechgen.github.io" style="color:white; font-style: italic;">Rusaid Ahamed</a></h2>
                </div>
                """

        st.markdown(html_temp, unsafe_allow_html=True)




if __name__ == '__main__':
    main()
