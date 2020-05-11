# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
def cluster_kmeans(df, k):
    """
    Clusters the m observations of n attributes 
    in the Pandas' dataframe df into k clusters.
    
    Euclidean distance is used as the proximity metric.
    
    Arguments:
        df   pandas dataframe of m rows with n columns (excluding index)
        k    the number of clusters to search for
        
    Returns:
        a m x 1 dataframe of cluster labels for each of the m observations
        retaining the original dataframe's (df's) index
        
        the final Sum-of-Error-Squared (SSE) from the clustering
    """
    # Sample fron the original df
    sample_df=df.sample(n = k)
    obs, attr= df.shape
    # Make copies 
    copy_df=df.copy()
    flag=0
    sse_old=0
    while (flag==0): 
        sse=0
        Labels=[]
        for i in range(0, obs):
            dist= []
            for j in range(0,k):
                #Calculate Eucledian distance
                diff=list((df.iloc[i,:]-sample_df.iloc[j,:])**2)
                eu_dist=(sum(diff))**(1/attr)
                dist.append(eu_dist) 
            #Add Labels to the observations based on the variable they are close to
            label=(dist.index(min(dist)))
            Labels.append(label)
            # Calculate SSE
            sse=sse+((min(dist) )**2)
        sse=sse**(1/2)
        copy_df['labels']=Labels
        # Stopping criteria is change in SSE should be 2 %
        if (sse_old !=0):
            if(abs(sse_old-sse)/sse_old<=0.05):
                flag=1 
                return_df=copy_df['labels'].to_frame()
                return (return_df, sse)
            else:
                sse_old=sse
                 #Empty the sample df
                sample_df.drop(sample_df.index, inplace=True)
                # Now pick random values from each label and add it to the sample df
                for val in range(0,k):
                    #Create new sample df
                    sample_df = pd.concat([sample_df, copy_df[copy_df['labels']==val].iloc[:,0:attr].sample(n=1)])
        else:
            sse_old=sse
            #Empty the sample df
            sample_df.drop(sample_df.index, inplace=True)
            for val in range(0,k):
                #Create new sample df 
                sample_df = pd.concat([sample_df, copy_df[copy_df['labels']==val].iloc[:,0:attr].sample(n=1)])