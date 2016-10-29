# ![Musebot](http://www.piggywheelapps.com/_/rsrc/1477261920406/home/ic_launcher.png) Musebot-ChaRNN
The vanilla Java character-level RNN which powers the [**Musebot**](https://play.google.com/store/apps/details?id=com.mran.textgenerator) app. This was ported from Andrej Karpathy's vanilla Python [RNN](https://gist.github.com/karpathy/d4dee566867f8291f086). numpy functionality is provided by an expansion of Princeton University's [Matrix](http://introcs.cs.princeton.edu/java/95linear/Matrix.java.html) and [StdRandom](http://introcs.cs.princeton.edu/java/95linear/StdRandom.java.html) classes. No libraries are used except for [Gson](https://github.com/google/gson).

## Features
- Create and train a language model on text input.
- Save checkpoints.
- Load in a model from a checkpoint and either continue training or run it.
