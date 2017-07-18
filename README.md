# A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues (VHRED) - Implementation in Tensorflow

<p align="center">
    <img src="https://github.com/lethienhoa/VHRED-implementation-in-Tensorflow/blob/master/Selection_057.png?raw=true" alt>
</p>
<p align="center">Variational Auto-Encoder for Dialogue Generation</p>

## Dataset

Ubuntu Dialogue Corpus (http://www.iulianserban.com/Files/UbuntuDialogueCorpus.zip)

This link contains a pre-processed versions of the Ubuntu Dialogue Corpus, which was used by Serban et al. (2016a) and Serban et al. (2016b) and originally developed by Lowe et al. (2015). There are three datasets: the natural language dialogues, the noun representations and the activity-entity representations. Each dataset is split into train, valid and test sets.

The task investigated by Serban et al. (2016a) and Serban et al. (2016b) is dialogue response generation. Given a dialogue context (one or several utterances), the dialogue model must generate an appropriate response (**a single utterance**).
The context and response pairs are provided (see "ResponseContextPairs" subdirectories).
The model responses by Serban et al. (2016a) and Serban et al. (2016b) are also provided (see "ModelPredictions" subdirectories).

## Reference Articles

A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues. Iulian Vlad Serban, Alessandro Sordoni, Ryan Lowe, Laurent Charlin, Joelle Pineau, Aaron Courville, Yoshua Bengio. 2016. http://arxiv.org/abs/1605.06069

Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models. Iulian V. Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, Joelle Pineau. 2016. AAAI. http://arxiv.org/abs/1507.04808.


## Reference Source Codes (Theano)

https://github.com/julianser/hed-dlg-truncated

--------------
MIT License
