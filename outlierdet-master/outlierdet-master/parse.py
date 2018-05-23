import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.decomposition import PCA # Principal Component Analysis module


mypath = '/root/activity2/files2'


def encode_observations(onlyfiles):
  listall = []
  maxlen = 0
  
  for f in onlyfiles:
      with open(f, 'r') as fl:
        for line in fl: 
          xlist = line.split(',')
          xlen = len(xlist)
          if xlen > maxlen:
             maxlen = xlen
          for xi in xlist:
             if xi not in listall:
               listall.append(xi)
  return listall,maxlen

def aencode(xlist, listall, maxcol):
    encod = []
    for i in range(0, maxcol):
      if i < len(xlist):
        encod.append(listall.index(xlist[i])) 
      else:
        encod.append(0.0) 
    return encod 

def create_array(listall, mfiles, cols):  

  linenum = 0
  for index, f in enumerate(mfiles):
      with open(f, 'r') as fl:
         for line in fl:
            linenum = linenum + 1
  total_rows = linenum
  print total_rows
  narray = np.empty([total_rows+1, cols])
  linenum = 0
  for index, f in enumerate(mfiles):
      with open(f, 'r') as fl:
         for line in fl:
           linenum = linenum + 1
           xlist = line.split(',')
           array_row = aencode(xlist, listall, cols)
           narray[ linenum ] = array_row
  print "NARRAY"
  print narray.shape
  return narray 
         

if __name__ == "__main__" :  
#   onlyfiles = [f for f in listdir(mypath)]
#   print onlyfiles
   onlyfiles = [f for f in listdir(mypath) if re.match('sha256.*process', f)]
   print "ONLY FILES"
   print onlyfiles 
   print len(onlyfiles) 
   mfiles = onlyfiles
   
   listall,maxcols = encode_observations(mfiles)
   print maxcols
   karray = create_array(listall, mfiles, maxcols)
   xx, yy = np.meshgrid(np.linspace(-11, 10, 500), np.linspace(-10, 5, 500))
   # Generate train data
#   X_train1,X_test1,X_outliers1 = karray[:50:], karray[50:60,:],karray[60:,:]
#   X_train1,X_test1,X_outliers1 = karray[:500:], karray[500:600,:],karray[600:,:]
   X_train1,X_test1,X_outliers1 = karray[:9000:], karray[9000:9010,:],karray[9010:9100,:]
   #PCA
   pca = PCA(n_components=2) 
   X_train = pca.fit_transform(X_train1)
   noise = np.random.normal(0, 1, X_train.shape)
   X_train = X_train + noise

   pca = PCA(n_components=2) 
   X_test = pca.fit_transform(X_test1)
   noise = np.random.normal(0, 1, X_test.shape)
   X_test = X_test + noise

   pca = PCA(n_components=2) 
   X_outliers = pca.fit_transform(X_outliers1)
   noise = np.random.normal(0, 1, X_outliers.shape)
   X_outliers = X_outliers + noise
   print X_train
   print X_test
   print X_outliers

   # fit the model
   clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
   clf.fit(X_train)
   y_pred_train = clf.predict(X_train)
   y_pred_test = clf.predict(X_test)
   y_pred_outliers = clf.predict(X_outliers)
   n_error_train = y_pred_train[y_pred_train == -1].size
   n_error_test = y_pred_test[y_pred_test == -1].size
   n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
   print(n_error_train)
   print(n_error_test)
   print(n_error_outliers)

   # plot the line, the points, and the nearest vectors to the plane
   Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)

   plt.title("Novelty Detection")
   plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
   a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
   plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

   s = 40
   b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
   b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
   c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
   plt.axis('tight')
#   plt.tight_layout()
   plt.xlim((-11, 5))
   plt.ylim((-10, 10))
   plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
   plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
   plt.show()
