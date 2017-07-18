# A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues (VHRED) - Implementation in Tensorflow

<p align="center">
    <img src="https://github.com/lethienhoa/VHRED-implementation-in-Tensorflow/blob/master/Selection_057.png?raw=true" alt>
</p>
<p align="center">Variational Auto-Encoder for Dialogue Generation</p>

## Dataset

Ubuntu Dialogue Corpus (http://www.iulianserban.com/Files/UbuntuDialogueCorpus.zip)

This link contains a pre-processed versions of the Ubuntu Dialogue Corpus, which was used by Serban et al. (2016a) and Serban et al. (2016b) and originally developed by Lowe et al. (2015). There are three datasets: the natural language dialogues, the noun representations and the activity-entity representations. Each dataset is split into train, valid and test sets.

The task investigated by Serban et al. (2016a) and Serban et al. (2016b) is dialogue response generation. Given a dialogue context (**one or several utterances**), the dialogue model must generate an appropriate response (**a single utterance**).
The context and response pairs are provided (see "ResponseContextPairs" subdirectories).
The model responses by Serban et al. (2016a) and Serban et al. (2016b) are also provided (see "ModelPredictions" subdirectories).

    *Context Examples:*
    1. anyone knows why my stock oneiric exports env var ' **unknown** I mean what is that used for ? I know of $USER but not $USERNAME . My precise install doesn't export USERNAME __eou__ __eot__ looks like it used to be exported by lightdm , but the line had the comment " // **unknown** : Is this required ?" so I guess it isn't surprising it is gone __eou__ __eot__ thanks ! How the heck did you figure that out ? __eou__ __eot__ https://bugs.launchpad.net/lightdm/+bug/864109/comments/3 __eou__ __eot__
    2. i set up my hd such that i have to type a passphrase to access it at boot . how can i remove that passwrd , and just boot up normal . i did this at install , it works fine , just tired of having reboots where i need to be at terminal to type passwd in . help ? __eou__ __eot__ backup your data , and re-install without encryption " might " be the easiest method __eou__ __eot__
    3. im trying to use ubuntu on my macbook pro retina __eou__ i read in the forums that ubuntu has a apple version now ? __eou__ __eot__ not that ive ever heard of .. normal ubutnu should work on an intel based mac . there is the PPC version also . __eou__ you want total control ? or what are you wanting exactly ? __eou__ __eot__
    4. no suggestions ? __eou__ links ? __eou__ how can i remove luks passphrase at boot . i dont want to use feature anymore ... __eou__ __eot__ you may need to create a new volume __eou__ __eot__ that leads me to the next question lol ... i dont know how to create new volumes exactly in cmdline , usually i use a gui . im just trying to access this server via usb loaded with next os im going to load , the luks pw is stopping me __eou__ __eot__ for something like that I would likely use something like a live gparted disk to avoid the conflict of editing from the disk __eou__ __eot__
    5. I just added a second usb printer but not sure what the uri should read - can anyone help with usb printers ? __eou__ __eot__ firefox localhost : 631 __eou__ __eot__ firefox ? __eou__ __eot__ yes __eou__ firefox localhost : 631 __eou__ firefox http://localhost:631 __eou__ cups has a web based interface __eou__ __eot__
    
    *Ground-truth respective responses examples:*
    1. nice thanks ! __eou__
    2. so you dont know , ok , anyone else ? __eou__ you are like , yah my mouse doesnt work , reinstall your os lolol what a joke __eou__
    3. just wondering how it runs __eou__
    4. you cant load anything via usb or cd when luks is running __eou__ it wont allow usb boot , i tried with 2 diff usb drives __eou__
    5. i was setting it up under the printer configuration __eou__ thanks ! __eou__
    
## Reference Articles

A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues. Iulian Vlad Serban, Alessandro Sordoni, Ryan Lowe, Laurent Charlin, Joelle Pineau, Aaron Courville, Yoshua Bengio. 2016. http://arxiv.org/abs/1605.06069

Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models. Iulian V. Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, Joelle Pineau. 2016. AAAI. http://arxiv.org/abs/1507.04808.


## Reference Source Codes (Theano)

https://github.com/julianser/hed-dlg-truncated

--------------
MIT License
