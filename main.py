# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
import pandas as pd
from PIL import Image

#import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import pickle


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

@st.cache
def directorySearch(folder,tag):
    results=list()
    results += [each for each in os.listdir(folder) if each.endswith(tag)]
    results=np.asarray(results)
    results=np.sort(results)
    return(results)

def main():
    global start_dens
    st.title("Multi Variate Contamination in Fuel")

    menu = ["Home", "Model Training", "Contaminant Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
        if image_file is not None:
            # To See Details
            # st.write(type(image_file))
            # st.write(dir(image_file))
            file_details = {"Filename": image_file.name, "FileType": image_file.type, "FileSize": image_file.size}
            st.write(file_details)

            img = load_image(image_file)
            st.image(img, width=250)

    if choice == "Model Training":
        if st.button("Start Train") :
            allResults = glob.glob('/batch31-38_with_target_diff_prx_2000tampered/*.csv',
                                   recursive=True)
            allResults = sorted(allResults, key=lambda x: (x.split("/")[-1]))
            #st.write(allResults)

            newpath1 = '/batch31-38_with_target_diff_prx_2000tampered/'
            # newpath1='/content/drive/MyDrive/OIL SAMPLES DATA1/'

            folder = newpath1  ## data directory
            tag = str('.csv')  ## format to import
            initString = '-'  ## string in csv file name to search for category (normal, sludge, water, together)
            fileList = directorySearch(folder, tag)
            # print(fileList)
            final_filelist = pd.DataFrame(index=range(0, len(fileList)),
                                          columns=['file', 'Target', 'file_dir', 'window_id'])
            for i in range(0, (len(fileList))):
                fileName = fileList[i]
                res1 = fileName.find(initString)
                if res1 == -1:
                    res1 = fileName.find('_')
                if res1 == -1:
                    print(res1)
                    res1 = 5
                c1 = int(res1 + 1)
                c5 = int(res1 + 12)

                wloc = fileName.rfind('W', c1, c5)
                sloc = fileName.rfind('S', c1, c5)
                tloc = fileName.rfind('T', c1, c5)
                finalCat = max([wloc, sloc, tloc])
                strCat = fileName[finalCat]
                # print(strCat)

                classLabel = int(0)
                if strCat == 'S':
                    final_filelist['file'][i] = fileName
                    final_filelist['Target'][i] = strCat
                    # print(fileName,'---Sludge')
                    classLabel = int(1)
                if strCat == 'W':
                    final_filelist['file'][i] = fileName
                    final_filelist['Target'][i] = strCat
                    # print(fileName,'---Water')
                    classLabel = int(2)
                if strCat == 'T':
                    final_filelist['file'][i] = fileName
                    final_filelist['Target'][i] = strCat
                    # print(fileName,'--- Mix')
                    classLabel = int(3)
                if strCat not in ['S', 'T', 'W']:
                    final_filelist['file'][i] = fileName
                    final_filelist['Target'][i] = strCat

                final_filelist['file_dir'][i] = allResults[i]
                final_filelist['window_id'][i] = i + 1


            ll = []
            for i, j in enumerate(final_filelist['file']):
                # print(i,j)
                head, tail = os.path.split(j)
                r1 = re.split('_', tail)
                r2 = re.split('-', r1[0])
                print(r2)
                # if len(r2)==3 and int(r2[1]) < 37 and int(r2[1])<37 and not  'A' in r1[0] :
                if len(r2) == 3 and 'A' not in r2[2]:
                    ll.append(tail)
                elif len(r2) == 2 and 'A' not in (r2[1]):
                    ll.append(tail)
                elif len(r2) == 4 and 'A' not in (r2[3]):
                    ll.append(tail)

            dff = pd.DataFrame({'file': ll})
            dff['file'].count()

            df4 = pd.DataFrame()
            c = 0
            # for i,j in enumerate(allResults):
            for i, j in enumerate(dff['file']):
                # print(i,j)
                df = pd.read_csv('/batch31-38_with_target_diff_prx_2000tampered/' + j)
                head, tail = os.path.split(j)
                # print(i,df.shape[1])
                df4[tail] = (df['Pressure_tmp'].rolling(300).std())

            df9 = pd.DataFrame(index=range(0, len(df4.columns)),
                               columns=['file', 'pre-trans_mean', 'trans_mean', 'post-trans_mean', 'transient_width'])

            for z, col in enumerate(df4.columns):
                start = 0
                end = 0

                a = df4[col]

                b = a.quantile(0.7)  # threshold set here : 70 percentile
                x = df4[col] > b  # find values greater than threshold

                # print(a)
                for i, j in enumerate(a):
                    # print(i,j)
                    if j > b:  # find value greater than threshold
                        start = i  # get the position of value greater than threshold
                        break
                for k, l in enumerate(a[start:]):  # now start checking from position that was marked earlier
                    # print(i,j)
                    if l < b and abs(
                            k) > 200:  # find values that are less than threshold and makesure check after 200 positions (for finding out better transient part)
                        end = start + k
                        break
                df9['file'][z] = col
                df9['pre-trans_mean'][z] = (df4[col].iloc[:start].mean())
                df9['trans_mean'][z] = (df4[col].iloc[start:end].mean())
                df9['post-trans_mean'][z] = (df4[col].iloc[end:].mean())
                if (end - start) > 0:
                    df9['transient_width'][z] = end - start
                else:
                    df9['transient_width'][z] = 0

            df5 = df4.describe().transpose()
            df5 = df5.reset_index()
            df10 = pd.merge(df9, df5[['index', 'std', 'max']], left_on='file', right_on='index', how='left')
            del df10['index']
            df10 = df10.set_index('file')

            df11 = pd.merge(df10, final_filelist[['file', 'Target']], left_on='file', right_on='file', how='left')
            df11 = df11.set_index('file')
            df11 = df11.astype({'pre-trans_mean': 'float64', 'trans_mean': 'float64', 'post-trans_mean': 'float64',
                                'transient_width': 'float64'})

            df12 = pd.DataFrame()
            for i, j in enumerate(dff['file']):
                # print(i,j)
                df = pd.read_csv('/batch31-38_with_target_diff_prx_2000tampered/' + j)
                head, tail = os.path.split(j)
                # print(i,df.shape[1])
                df12[tail] = (df['Density'].rolling(300).std())

            df13 = pd.DataFrame(index=range(0, len(df12.columns)),
                                    columns=['file', 'pre-trans_mean-density', 'trans_mean-density',
                                             'post-trans_mean-density', 'transient_width-density'])

            for z, col in enumerate(df12.columns):
                start = 0
                end = 0
                print(col)  # file name
                a = df12[col]

                b = a.quantile(0.7)  # threshold set here : 70 percentile
                x = df12[col] > b  # find values greater than threshold
                # print(a)
                for i, j in enumerate(a):
                    # print(i,j)
                    if j > b:  # find value greater than threshold
                        start = i  # get the position of value greater than threshold
                        break
                for k, l in enumerate(a[start:]):  # now start checking from position that was marked earlier
                    # print(i,j)
                    if l < b and abs(
                            k) > 200:  # find values that are less than threshold and makesure check after 200 positions (for finding out better transient part)
                        end = start + k
                        break
                df13['file'][z] = col
                df13['pre-trans_mean-density'][z] = (df12[col].iloc[:start].mean())
                df13['trans_mean-density'][z] = (df12[col].iloc[start:end].mean())
                df13['post-trans_mean-density'][z] = (df12[col].iloc[end:].mean())
                if (end - start) > 0:
                    df13['transient_width-density'][z] = end - start
                else:
                    df13['transient_width-density'][z] = 0
            df13 = df13.astype(
                {'pre-trans_mean-density': 'float64', 'trans_mean-density': 'float64', 'post-trans_mean-density': 'float64',
                 'transient_width-density': 'float64'})
            df11.drop(['std'], axis=1, inplace=True)

            df14 = df13[['file', 'pre-trans_mean-density', 'post-trans_mean-density']]
            df14['pre-trans_mean-density'] = df14['pre-trans_mean-density'].fillna(0)

            df11.dropna(inplace=True)

            le = preprocessing.LabelEncoder()
            df11['Target'] = le.fit_transform(df11['Target'])
            df11.loc[:, 'Target']

            df15 = df11.merge(df14, how='inner', on='file')
            del df15['file']
            df15 = df15[['pre-trans_mean', 'trans_mean', 'post-trans_mean', 'transient_width', 'max', 'pre-trans_mean-density',
                 'post-trans_mean-density', 'Target']]
            st.write(df15)
            col = df15.columns

            features = col.tolist()
            feature = features[:-1]
            target = features[-1]

            # x=dff_tr.loc[:,feature].values
            # y=dff_tr.loc[:,target].values
            x = df15.loc[:, feature].values
            y = df15.loc[:, target].values
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=98)
            ost = SMOTE()
            os_data_X, os_data_y = ost.fit_resample(x_train, y_train)
            os_data_X = pd.DataFrame(data=os_data_X, columns=feature)
            os_data_y = pd.DataFrame(data=os_data_y, columns=['Target'])

            # print('After Oversampling:')
            os_data_X, os_data_y = ost.fit_resample(x_train, y_train)
            clf_rf_bal = RandomForestClassifier(n_estimators=10, random_state=99)
            clf_rf_bal = clf_rf_bal.fit(os_data_X, os_data_y)

            #from sklearn.inspection import permutation_importance
            #results = permutation_importance(clf_rf_bal, x, y, scoring='accuracy')
            #importance = results.importances_mean
            # summarize feature importance
            #print('using permutaiton feature importance')
            #for i, v in enumerate(importance):
            #    print('Feature: %0d, Score: %.5f' % (i, v))
            #importance = clf_rf_bal.feature_importances_
            # summarize feature importance
            #print('using feature importance')
            #for i, v in enumerate(importance):
            #    print('Feature: %0d, Score: %.5f' % (i, v))

            bal_cm = confusion_matrix(y_test, clf_rf_bal.predict(x_test))
            y_pred_bal = clf_rf_bal.predict(x_test)

            print('balanced classification report')
            cls_rpt=classification_report(y_test, y_pred_bal)
            st.write(f'classification report : {cls_rpt}')

            bal_ac = accuracy_score(y_test, clf_rf_bal.predict(x_test))
            st.write(f'accuracy score : {bal_ac}')

            filename = 'finalized_model1.pkl'
            pickle.dump(clf_rf_bal, open(os. path. join(os.getcwd(), filename), 'wb'))




    if choice == "Contaminant Prediction":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if st.button("Process") and data_file is not None:
            file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
            st.write(file_details)

            df = pd.read_csv(data_file)
            st.dataframe(df)

            tag = str('.csv')  ## format to import
            initString = '-'  ## string in csv file name to search for category (normal, sludge, water, together)
            fileName = data_file.name
            # print(fileList)
            final_filelist = pd.DataFrame(columns=['file', 'Target'])
            res1 = fileName.find(initString)
            if res1 == -1:
                res1 = fileName.find('_')
            if res1 == -1:
                print(res1)
                res1 = 5
            c1 = int(res1 + 1)
            c5 = int(res1 + 12)

            wloc = fileName.rfind('W', c1, c5)
            sloc = fileName.rfind('S', c1, c5)
            tloc = fileName.rfind('T', c1, c5)
            finalCat = max([wloc, sloc, tloc])
            strCat = fileName[finalCat]

            st.write(f'FileName:{fileName}')
            if strCat not in ['S', 'T', 'W']:
                strCat = 'No Contaminant'
                st.write('No Contaminant')
            if strCat in ['S', 'T', 'W']:
                st.write('Contaminant Exists')
            if strCat == 'S':
                st.write('Type of Contaminant: Sludge')
            if strCat == 'W':
                st.write('Type of Contaminant: Water')
            if strCat == 'T':
                st.write('Type of Contaminant: Sludge+Water')

            df4 = pd.DataFrame()
            df4['roll_std'] = df['Pressure_tmp'].rolling(300).std()
            df5 = df4.describe().transpose()
            df5 = df5.reset_index()
            maxx = df5['max'][0]
            df9 = pd.DataFrame(
                columns=['file', 'pre_trans_mean', 'trans_mean', 'post_trans_mean', 'transient_width'])

            # for col in df4.columns:
            # end = 0
            # print(col)  # file name
            # a = df4[col]
            a = df4['roll_std']
            st.write(a)

            b = a.quantile(0.7)  # threshold set here : 70 percentile
            # print(b)
            st.write(b)
            # x = df4[col] > b
            x = df4['roll_std'] > b  # find values greater than threshold
            # print(x.value_counts())
            # print(a)
            st.write(x)
            for i, j in enumerate(a):
                # print(i,j)
                if j > b:  # find value greater than threshold
                    start = i  # get the position of value greater than threshold
                    break
            for k, l in enumerate(a[start:]):  # now start checking from position that was marked earlier
                # print(i,j)
                if l < b and abs(
                        k) > 200:  # find values that are less than threshold and makesure check after 200 positions (for finding out better transient part)
                    end = start + k
                    break

            file = data_file.name
            pre_trans_mean = (df4['roll_std'].iloc[:start].mean())
            trans_mean = (df4['roll_std'].iloc[start:end].mean())
            post_trans_mean = (df4['roll_std'].iloc[end:].mean())
            if (end - start) > 0:
                transient_width = end - start
            else:
                transient_width = 0


            df12 = pd.DataFrame()
            df12['roll_std_den'] = (df['Density'].rolling(300).std())

            df13 = pd.DataFrame()
            df13 = pd.DataFrame(columns=['file', 'pre_trans_mean_dens', 'trans_mean_dens', 'post_trans_mean_dens',
                                         'transient_width_dens'])

            # for col in df4.columns:
            # end = 0
            # print(col)  # file name
            # a = df4[col]
            p = df12['roll_std_den']

            q = p.quantile(0.7)  # threshold set here : 70 percentile
            # print(b)
            # st.write(b)
            # x = df4[col] > b
            xx = df12['roll_std_den'] > q  # find values greater than threshold
            # print(x.value_counts())
            # print(a)
            # st.write(xx)
            for i, j in enumerate(p):
                # print(i,j)
                if j > q:  # find value greater than threshold
                    start_dens = i  # get the position of value greater than threshold
                    break
            for k, l in enumerate(p[start_dens:]):  # now start checking from position
                # print(i,j)
                if l < q and abs(k) > 200:  # find values that are less than threshold
                    end_dens = start_dens + k
                    break

            pre_trans_mean_dens = df12['roll_std_den'].iloc[:start_dens].mean()
            trans_mean_dens = df12['roll_std_den'].iloc[start_dens:end_dens].mean()
            post_trans_mean_dens = df12['roll_std_den'].iloc[end_dens:].mean()
            if (end_dens - start_dens) > 0:
                transient_width_dens = end_dens - start_dens
            else:
                transient_width_dens = 0

            zz = {'file': file, 'pre_trans_mean': pre_trans_mean, 'trans_mean': trans_mean,
                  'post_trans_mean': post_trans_mean,
                  'pre_trans_mean_dens': pre_trans_mean_dens, 'trans_mean_dens': trans_mean_dens,
                  'post_trans_mean_dens': post_trans_mean_dens
                  }

            pre_trans_mean1 = pre_trans_mean
            trans_mean1 = trans_mean
            post_trans_mean1 = post_trans_mean
            transient_width1 = transient_width
            max1 = maxx
            pre_trans_mean_dens1 = pre_trans_mean_dens
            post_trans_mean_dens1 = post_trans_mean_dens

            st.write(zz)
        # load the model from disk
            loaded_model = pickle.load(open('finalized_model1.pkl', 'rb'))

            result = loaded_model.predict([[pre_trans_mean1,trans_mean1,post_trans_mean1,transient_width1,max1,
                                     pre_trans_mean_dens1,post_trans_mean_dens1]])

            if result==0:
                 st.write(f'Predicted Contaminant: Sludge')
            if result==1:
                st.write(f'Predicted Contaminant: Water')
            if result==2:
                st.write(f'Predicted Contaminant: Water+Sludge')
            if result==3:
                st.write('No Contaminant')

if __name__ == "__main__":
    main()
