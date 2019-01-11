# Pre Routines
Preprocess - >  Load data, parsing (into iterator of objects)
Boot Camp -> (init with features) iterable of sentence objects -> to feature tensor

Models ->

#Train
features - > model
model fit  

#Predict
new sentence - > pre routines-> data obj with feature tensor 
-> model predict (Chu Liu)
 



Sentence_object:
    sentence_object.graph
    node_object.tags  # only used for f creation
    node_object.words  # only used for f creation
    node_object.f
     