// src/data/blogItems.ts
export interface BlogItem {
  id: string;
  slug: string;
  image: string;
  alt: string;
  tag: string;
  date: string;
  title: string;
  description: string;
  href: string;
  color: 'neon-cyan' | 'neon-purple';
  readTime: string;
  content: string; // HTML content for blog detail
  author: {
    name: string;
    role: string;
    bio: string;
    image: string;
    alt: string;
  };
  relatedArticles: { id: string }[]; // References to other blog items by id
}

export const blogItems: BlogItem[] = [
  {
    id: 'webgl',
    slug: 'transformers',
    image: '../img/CNNTrans/Transformers.jpg',
    alt: 'WebGL Blog',
    tag: 'AI',
    date: 'march 25, 2025',
    title: 'From Loops to Lightning - How Transformers Outran RNNs',
    description:
      'This article gives an in-depth overview of the Transformer architecture, which has revolutionized natural language processing. It focuses on attention blocks, the key component of the model that establishes parallel and contextual connections between words in a sentence',
    href: '/blogs/transformers',
    color: 'neon-cyan',
    readTime: '5 min read',
    content: `
      <h1><strong>Transformers</strong>: The Evolution Beyond <strong>RNNs</strong></h1>
      <p>Before the advent of <strong>Transformers</strong>, <strong>Recurrent Neural Networks (RNNs)</strong> were the go-to models for handling <strong>sequential data</strong>, where the order of elements plays a crucial role — as is the case in <strong>natural language processing</strong>, <strong>time series</strong>, and other domains. Inspired by traditional <strong>feedforward neural networks</strong>, <strong>RNNs</strong> introduced the ability to process data step by step, maintaining a form of <strong>memory</strong> across the sequence.</p>
      <p>However, this <strong>sequential approach</strong> comes with significant limitations. First, <strong>RNNs</strong> process inputs one element at a time, preventing full utilization of modern <strong>GPUs</strong>, which are designed for <strong>parallel computation</strong> — this makes training relatively slow. Second, <strong>RNNs</strong> struggle to capture <strong>long-range dependencies</strong> within sequences. As information is passed from step to step, it tends to degrade or vanish, especially over long distances, leading to what is commonly known as the <strong>vanishing gradient problem</strong>.</p>
      <p>It is in this context that <strong>Transformers</strong> revolutionized the field. While inspired by the <strong>encoder-decoder</strong> frameworks of <strong>RNNs</strong>, <strong>Transformers</strong> remove the notion of <strong>recurrence</strong> entirely, replacing it with a fully <strong>Attention-based mechanism</strong>. This allows the model to focus directly on the most relevant parts of a sequence, regardless of their position.</p>
      <p>With this innovation, <strong>Transformers</strong> not only surpassed <strong>RNNs</strong> in performance on key <strong>NLP</strong> tasks — such as <strong>machine translation</strong>, <strong>text summarization</strong>, and <strong>speech recognition</strong> — but also unlocked new applications across various domains, including <strong>computer vision</strong> and <strong>bioinformatics</strong>.</p>
      <p>So why did <strong>Transformers</strong> replace <strong>RNNs</strong>? Because they directly address the two critical limitations of <strong>recurrent models</strong>:</p>
      <ul>
        <li>They enable <strong>parallel processing</strong> of sequence data, significantly speeding up training.</li>
        <li>They effectively capture <strong>long-range dependencies</strong> through the <strong>Attention mechanism</strong>.</li>
      </ul>
      <p>In short, the rise of <strong>Transformers</strong> represents a natural and necessary evolution beyond the structural limitations of <strong>RNNs</strong>.</p>
      <p>Next, let’s dive deeper into how this groundbreaking architecture works.</p>

      <h2>The <strong>Transformer Architecture</strong></h2>
      <p><img src="../public/img/CNNTrans/transfArchi.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;" /></p>
      <h3>Overview</h3>
      <p>Originally designed for tasks like <strong>machine translation</strong>, <strong>Transformers</strong> are highly effective at converting input sequences into output sequences. They are the first model to rely entirely on the <strong>self-attention mechanism</strong>, without using <strong>RNNs</strong> or <strong>convolutional networks</strong>. However, they still maintain the classic <strong>encoder-decoder</strong> structure.</p>
      <p>If we think of the <strong>Transformer</strong> as a black box for translation, it takes a sentence in one language — say, English — and translates it into another language, like Spanish.</p>
      <p>Now, if we take a closer look at this black box, it consists of two main parts:</p>
      <ul>
        <li>The <strong>encoder</strong> takes the input (for example, "How are you?") and transforms it into a <strong>matrix representation</strong>.</li>
        <li>The <strong>decoder</strong> uses this encoded information to gradually generate the translated sentence, such as "Comment allez vous?".</li>
      </ul>
      <p>In reality, both the <strong>encoder</strong> and <strong>decoder</strong> are made up of multiple stacked layers, all with the same structure. Each <strong>encoder layer</strong> processes the input and passes it to the next one. On the <strong>decoder</strong> side, each layer takes input from both the last <strong>encoder layer</strong> and the previous <strong>decoder layer</strong>.</p>
      <p>In the original <strong>Transformer</strong> model, there were 6 layers for the <strong>encoder</strong> and 6 layers for the <strong>decoder</strong>, but this number (N) can be adjusted as needed.</p>
      <p>Now that we have a general idea of the Transformer architecture, let’s dive deeper into how the encoders and decoders work.</p>
      <p><img src="../public/img/CNNTrans/encDecWorflow.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;" /></p>

      <h2><strong>Encoder Workflow</strong></h2>
      <p><img src="../public/img/CNNTrans/encworkflow.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;" /></p>
      <p><strong>Transformers</strong> are currently the basis for many advances in <strong>natural language processing (NLP)</strong>. They have enabled models such as <strong>BERT</strong>, <strong>GPT</strong>, and <strong>T5</strong> to achieve impressive results. At the heart of these models is the <strong>Transformer encoder</strong>, which transforms input data (such as sentences) into <strong>machine-readable information</strong>.</p>

      <p>In this section, we will simply explain the key elements that make up this encoder: <strong>embeddings</strong> (which transform words into numbers), <strong>positional coding</strong> (which indicates word order), <strong>self-attention</strong> (which allows the model to know which words are important), <strong>normalization</strong> and <strong>residual connections</strong> (which help stabilize learning), and finally the <strong>feed-forward layer</strong> (which processes the information before moving on to the next step).</p>

      <p>Each of these elements is <strong>essential</strong> for the Transformer to perform so well on tasks such as <strong>translation</strong>, <strong>summarization</strong>, and <strong>text comprehension</strong>.</p>

      <h3>1. Input Embeddings</h3>
      <p>In a Transformer, the very first step is to convert <strong>words into numbers</strong>. Computers don’t directly understand words or sentences like we do; they need numbers to work. This is where <strong>input embedding</strong> (or “vectorizing words”) comes in. This step is only done in the very first encoder (often called the <strong>“bottom encoder”</strong>).</p>

      <p><strong>How it works:</strong></p>
      <ul>
          <li>We take the sentence as input: for example, <em>“How are you?”</em></li>
          <li>This sentence is split into <strong>tokens</strong> (i.e., words or pieces of words). Example: “How”, “are”, “you”.</li>
          <li>Each token is transformed into a <strong>vector</strong>, e.g., [0.25, -0.14, 0.67, …], via an <strong>embedding layer</strong>.</li>
          <li>These vectors are not random; they <strong>capture word meaning</strong>. For example, “king” will be closer to “queen” than to “apple” in this mathematical space.</li>
          <li><strong>The size of each vector is always the same</strong>: in basic Transformers, each vector has 512 dimensions (that is, 512 numbers inside, regardless of the word).</li>
      </ul>

      <p><img src="../public/img/CNNTrans/encworkflowinput.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;" /></p>
      <h3>2. Positional Encoding</h3>
      <img src="../public/img/CNNTrans/positionalenc.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">
      <p>Transformers, the artificial intelligence models revolutionizing language processing, have a unique characteristic: unlike their predecessors like recurrent neural networks (RNNs), they don’t process the words of a sentence one by one. Instead, they analyze all the words simultaneously, which allows them to be much faster.</p>

      <p>However, this approach poses a problem: how can the model understand the order of the words? After all, the meaning of a sentence crucially depends on the order in which the words are placed. For example, “The cat sleeps on the mat” has a very different meaning from “The mat sleeps on the cat!” Without information about the order, the Transformer could confuse these two sentences.</p>

      <p><em><strong>An Ingenious Trick</strong></em></p>
      <p>To solve this problem, researchers invented a technique called “positional encoding.” The idea is simple: we add information to each word that indicates its position in the sentence. It’s a bit like giving each word an address.</p>

      <p><em><strong>How it Works: Mathematical Waves to Encode Position</strong></em></p>
      <p>Instead of using simple numbers to encode the position, we use mathematical waves, specifically sine and cosine functions. Each position in the sentence receives a unique “mathematical signature” created from these waves.</p>
      <p>Why use sines and cosines? These functions have interesting properties:</p>
      <ul>
          <li><strong>They are periodic</strong>: They repeat, which is convenient for long sentences.</li>
          <li><strong>They are different</strong>: We use waves of different frequencies (faster or slower) to encode each position. Thus, each position has a unique signature.</li>
      </ul>

      <p><em><strong>In Practice: The Magic Addition</strong></em></p>
      <ul>
          <li><strong>Word Representation</strong> (Embeddings): Each word in the sentence is first transformed into a vector of numbers called an “embedding.” This vector represents the meaning of the word.</li>
          <li><strong>Encoding the Position</strong>: For each position in the sentence, we calculate a vector of numbers using sine and cosine functions. This vector represents the position of the word.</li>
          <li><strong>Combining the Two</strong>: We add the word’s embedding vector and the position vector</li>
      </ul>
      <p>Thanks to this addition, each word now has a representation that combines:</p>
      <ul>
          <li><strong>Its meaning</strong>: The word’s embedding.</li>
          <li><strong>Its position</strong>: The position vector encoded with sine and cosine waves.</li>
      </ul>

      <h3>3. Stack of Encoder Layers</h3>
      <img src="../public/img/CNNTrans/stackenc.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">
      <p>Each <strong>encoder layer</strong> contains:</p>

      <p>a) <strong>Multi-Head Self-Attention</strong>:<br>
      When a machine wants to understand or translate a sentence, the encoder is the main tool that reads and analyzes the sentence. It’s a bit like the model’s brain, trying to grasp the overall meaning before taking action.</p>

      <p>The encoder uses a key method: self-attention. Imagine that each word in the sentence looks around and asks itself:“What other words help me better understand the meaning of the sentence?”</p>
      <p>For example, in the sentence “You’re nice,” the word “es” will quickly understand that it is linked to “Tu,” because together they form the meaning of the action. This ability to make connections is crucial for properly grasping the context.</p>

      <p>To make these connections, each word is transformed into three elements:</p>
      <ul>
          <li><strong>Query</strong>: each word poses a question to other words, such as: “Who can help me understand?”</li>
          <li><strong>Key</strong>: this is the identity of each word, a sort of badge that says: “This is who I am.”</li>
          <li><strong>Value</strong>: this is the information the word carries and can share if needed.</li>
      </ul>
      <p>👉 When a word poses its question (Query) and another word has the correct key (Key), then the answer is positive: we keep the Value of the found word to enrich the overall understanding.</p>
      <p>And that’s not all! The encoder doesn’t just look at a single sentence. They repeat this process several times in parallel, slightly changing the way they ask questions or read the keys.</p>
      <p>It’s as if several people were looking at the same sentence but from different perspectives: some will focus on the grammar, others on the tone or the cause-and-effect relationships.</p>

      <p>b) <strong>Normalization and Residual Connections</strong>:<br>
      <img src="../public/img/CNNTrans/normaworkflow.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;"></p>
      <p>Once the encoder has completed the self-attention phase (where the words have looked at the other words to capture the context), we don’t move directly to the next step. Before that, two important operations are performed:</p>

      <p>— <strong>The residual connection (or “shortcut”)</strong><br>
      Imagine that the result of the self-attention is like an enriched version of what the model has understood about the relationships between words.<br>
      This result is added to the original input, meaning the representations of the words before self-attention.</p>
      <p>👉 Why?<br>
      Because it helps preserve the initial information intact and adds on top of what self-attention has discovered.<br>
      It’s a bit like having a draft with ideas and highlighting the important parts without ever erasing your original text.</p>
      <p>This is called a <strong>residual connection</strong>: we add the “residue” (the original input) to the layer’s result.</p>

      <p>— <strong>Normalization</strong><br>
      Next, this sum passes through a normalization (called Layer Normalization).<br>
      This step serves to stabilize the values to prevent them from becoming too large or too small.</p>
      <p>👉 Why?<br>
      It helps the model to learn better during training and avoids certain problems like forgetting or losing information (the famous “vanishing gradient” problem).</p>
      <p>⚠️ This duo (residual + normalization) is a ritual in the encoder of Transformers: after every sub-layer (whether self-attention or feed-forward neural network), it is repeated.</p>


      <p>c) <strong>Feed-Forward Neural Network</strong>:<br>
      <img src="../public/img/CNNTrans/feedforward.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;"></p>
      <p>After the information passes through self-attention and the residual connection, it enters a new phase called the feed-forward neural network. Here’s how it works:</p>
      <p><strong>Information Transformation:</strong><br>
      The information first goes through a series of two simple transformations, called linear layers. These layers modify the information, kind of like changing its shape.</p>
      <p><strong>ReLU Activation:</strong><br>
      Next, there’s a small “filter” called ReLU. This filter helps make the information clearer and more useful for the next steps. It’s like highlighting important details to make them stand out.</p>
      <p><strong>Return to the Input:</strong><br>
      The transformed information is added back to the original information (this is the residual connection). This helps keep the original idea intact while adding new insights, without losing any important elements.</p>
      <p><strong>Normalization:</strong><br>
      Finally, everything goes through a last step called normalization. This step adjusts the information to ensure everything is balanced before moving on to the next phase.</p>


      <h3>4. Output of the encoder</h3>
      <p>The output of the final layer of the encoder is a set of vectors. Each of these vectors represents the input sequence, but with a rich and deep contextual understanding. These vectors contain all the information processed by the encoder, capturing the relationships and meaning of the words in the sequence.</p>

      <p>This encoded output is then passed to the decoder in the Transformer model. The decoder uses these vectors to generate predictions or output sequences, such as translating text or producing a response. The encoder’s job is to prepare this information so that the decoder can focus on the correct parts of the input when decoding.</p>

      <p>Think of the encoder’s function as building a tower made of layers. You can stack multiple encoder layers, and each layer adds more understanding by looking at the input from a slightly different perspective. Each layer uses its own attention mechanism, allowing it to learn different aspects of the input. As you stack layers, you increase the depth and richness of understanding. This process helps improve the transformer’s ability to predict and generate outputs more accurately, refining understanding layer by layer.</p>


      <h2><strong>Decoder Workflow</strong></h2>
      <img src="../public/img/CNNTrans/decoderworkflow.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

      <p>At the heart of the Transformer model lies the decoder, whose mission is to generate coherent text sequences. Like the encoder, it is made up of several sub-layers, including multi-head attention mechanisms, neural networks, and normalization techniques. These components work together to decode the information prepared by the encoder and produce intelligible text.</p>

      <p>The decoder operates in an autoregressive manner, meaning it generates text step by step, based on the words it has already produced. It starts with a seed symbol, then uses the information from the encoder, enriched by attention, to select the next word. This process continues sequentially, with each new word influenced by the previous ones and by the context provided by the encoder. The decoder stops when it generates an end symbol, signaling the completion of the text sequence. In short, the decoder is an orchestrator that transforms encoded information into fluid and relevant text.</p>

      <h3>1. Output</h3>
      <p>Unlike the encoder, which receives the input sentence directly, the decoder requires a starting point to initiate text generation. This starting point is the famous <strong>“Outputs,”</strong> which feeds the <strong>“Output Embedding”</strong> at the very beginning of the decoding process. This first “Outputs” comes from the Special <strong>“Start of Sequence” (SOS) Token</strong>, a special token added to the model’s vocabulary. This token is not part of the common words of the language, such as nouns, verbs, or adjectives. Its main role is to signal to the decoder that a new text sequence will be generated, thus acting as a start signal. This SOS Token is therefore the very first “Outputs” sent to the “Output Embedding” layer to start the generation process.</p>

      <h3>2. Output Embeddings</h3>
      <p>The <strong>“Output Embedding”</strong> is a matrix, or a lookup table, that associates each token in the vocabulary, including the SOS token, with a numerical vector. This vector makes it possible to represent each token in a multidimensional space, where tokens with similar meanings are positioned close to each other. Concretely, when the SOS token is sent to the Output Embedding, the matrix extracts the vector corresponding to this token. This vector then becomes the initial input to the decoder. It is this vector that will be combined with the positional encoding and processed by the different layers of the decoder, such as masked attention, multi-head attention, etc., to progressively generate the text sequence.</p>

      <h3>3. Positional Encoding</h3>
      <p>Positional encoding is a crucial step in the sequence processing process in a Transformer model, whether in the encoder or the decoder. This step comes after token embedding, and its role is to add information about the position of words in the sequence.</p>
      <p>Transformer models do not process sequences sequentially, like RNNs or LSTMs, which process each word one by one in a specific order. Instead, the words are processed in parallel. However, this poses a problem: without an indication of their position, the model would not know whether the word “cat” comes before or after “eat.” This is where positional embeddings come in.</p>
      <p>Positional embeddings add a unique vector to each word in the sequence to indicate its specific position. For example, the first word in the sequence could have a different positional encoding than the second word, and so on. These vectors are added to the word embeddings, allowing the model to maintain an understanding of word order.</p>
      <p>It works like this: each token in the input sequence is associated with a vector representing its position, and then this positional encoding is added to the vector of the corresponding word. This allows the model to “know” the order in which the words appear in the sentence, and therefore better understand the context.</p>
      <p>This process is identical in both the encoder and the decoder.</p>

      <h3>4. Stack of Decoder Layers</h3>
      <p>🔔 Important context before we begin:</p>
      <p>We want to train a model for translate englsh sentence to french. During training, the complete French sentence is already available (because it’s part of the data the model needs to learn). But to prevent the decoder from “cheating” by looking at future words during training, we use a mask in the self-attention process. This mask is like a filter that prevents the model from seeing subsequent words in the sentence while it learns to make predictions.</p>

      <p>a) <strong>Masked Self-Attention</strong>:<br>
      <img src="../public/img/CNNTrans/multiatten.jpg" alt="architecture"><br>
      🔍 <strong>Role during training:</strong><br>
      When the decoder learns to generate the French sentence, it must only use the words already generated (on the left) and not see future words (on the right).</p>
      <p>➡️ <strong>Concrete example:</strong><br>
      Let’s imagine we’re learning to generate: “The cat was sitting on the carpet”</p>
      <p>When the model has to learn to predict the word “sitting”, it can only use:</p>
      <ul>
          <li>“The”</li>
          <li>“cat”</li>
          <li>“was”</li>
      </ul>
      <p>And cannot see subsequent words like “on” or “the carpet” even if they are available in the example.</p>
      <p>🛑 <strong>Why?</strong><br>
      Because in real life, when generating text, we never have future words in advance.</p>
      <p>💡 <strong>How does it work technically?</strong><br>
      We apply a triangular mask (called a causal mask) to the attention matrix. This mask places “-∞” on the attention scores of future words.<br>
      Result: the decoder learns to look only at words already known to predict the next one.</p>
      <p>🎨 <strong>Simple analogy:</strong><br>
      It’s like writing a sentence without cheating: when you write the word “assis,” you only look at what you just wrote before, not what comes after.</p>

      <p>b) <strong>Encoder-Decoder Multi-Head Attention or Cross Attention</strong><br>
      🔄 <strong>What’s changing here:</strong><br>
      Once the masked self-attention is complete, the decoder will look for clues in the encoded English sentence.</p>
      <p>➡️ <strong>Concrete example:</strong><br>
      Still for the word “assis” to be generated, the decoder will consult the encoded English sentence “The cat sat on the mat”.</p>
      <p>It will ask itself:</p>
      <p><strong>“Where is the key information in the English?”</strong></p>
      <p>Here, it will notice that the word “sat” is important because it is the verb to be translated as “assis”.</p>
      <p>🧠 <strong>Where does this information come from?</strong><br>
      The encoder’s outputs, which are available for ALL English words from the start (because the English sentence is encoded in its entirety before starting the decoder).</p>
      <p>⚠️ <strong>But be careful:</strong><br>
      The decoder still looks ONLY at the words already generated on the French side, thanks to the masked self-attention seen just above. But for these words, he can use ALL English words via the encoder to better understand them.</p>
      <p>🎨 <strong>Simple analogy:</strong><br>
      It’s as if you were translating sentence by sentence:<br>
      You write “assis” on the French side, and each time you look at the complete English sentence to check how to translate it correctly.</p>

      <p>b) <strong>Feed-Forward Neural Network</strong><br>
      ⚙️ <strong>Last step of a decoder layer:</strong><br>
      Here, each word processed by attention is then refined.<br>
      The feed-forward network will transform the word representation by applying two linear layers and a non-linear activation.</p>
      <p>➡️ <strong>Example:</strong><br>
      The internal representation of the word “assis” is improved to be ready to move on to the next step (e.g., predicting the next word “sur”).</p>
      <p>🎨 <strong>Simple analogy:</strong><br>
      It’s as if, after choosing the word “assis,” you further refine your idea by adding details and nuances before writing the next word.</p>

      <p>c) <strong>Linear Classifier and Softmax for Generating Output Probabilities</strong><br>
      After passing through all the decoder layers (masked self-attention, encoder-decoder attention, feed-forward network), we obtain a vector representation for the word the model is about to generate. But at this stage, it’s still just an abstract vector, with no word assigned yet.</p>
      <p>🎯 <strong>What happens next?</strong></p>
      <ul>
      <li><strong>Linear Classifier:</strong><br>This vector is fed into a linear classifier (a linear layer).<br>The classifier transforms the vector into a list of scores, one score for each word in the vocabulary.<br><strong>Example:</strong> If the model knows 10,000 words, it will produce 10,000 scores (one for “sat,” another for “jumped,” etc.).</li>
      <li><strong>Softmax:</strong><br>The softmax function takes these raw scores and turns them into <strong>probabilities</strong>.<br>Each word gets a probability representing how likely it is to be the correct word in this context.<br><strong>Example:</strong><br>“sat” : 65%<br>“jumped” : 20%<br>“ran” : 10%<br>The remaining words will have smaller probabilities.</li>
      <li><strong>Selecting the final word:</strong><br>The model picks the word with the highest probability (e.g., “sat” at 65%).<br>This word becomes the newly generated word, and will be used as input for the next step to predict the following word.<br><strong>Simple analogy:</strong><br>It’s like playing a guessing game with multiple choices:<br>You look at all the possible options (all 10,000 words),<br>You think: “Hmm… 65% chance it’s ‘sat,’ 20% for ‘jumped’…”<br>Then you choose the most likely answer.</li>
      </ul>

      <img src="../public/img/CNNTrans/transoutput.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

      <h2>Why Transformers Changed Everything ?</h2>

      <p>The Transformer architecture has fundamentally reshaped the landscape of natural language processing (NLP) and sequence modeling.</p>
      <p>Unlike Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs), which process sequences token by token in order, Transformers introduced parallel processing thanks to their attention mechanisms. This shift allowed models to capture long-range dependencies much more efficiently, without being limited by sequential bottlenecks.</p>

      <h3>🚀 Key breakthroughs of Transformers over RNNs:</h3>
      <ul>
          <li><strong>Full attention span</strong>: Transformers can directly “attend” to every word in a sentence at once, while RNNs struggle with remembering distant words due to vanishing gradients.</li>
          <li><strong>Faster training</strong>: By processing all tokens in parallel during training (in the encoder), Transformers drastically reduce computation time compared to the step-by-step nature of RNNs.</li>
          <li><strong>Scalability</strong>: Transformers easily scale to large datasets and massive model sizes (e.g., GPT, BERT), unlocking unprecedented performance on tasks like translation, summarization, and text generation.</li>
      </ul>

      <h3>🧠 A smarter architecture:</h3>
      <p>By combining <strong>self-attention, masked attention, cross-attention, and feed-forward layers</strong>, the Transformer builds highly contextualized word representations, making it a universal tool not just for NLP, but also for vision, audio, and multi-modal tasks.</p>
      <p>Today, almost every state-of-the-art model — from ChatGPT to BERT — relies on the Transformer backbone, making it one of the most influential breakthroughs in AI history.</p>
    `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Mark Thomson',
    },
    relatedArticles: [{ id: '3d' }, { id: 'ar' }],
  },
  {
    id: 'ai',
    slug: 'rnn',
    image: '../img/CNNTrans/blog2.jpg',
    alt: 'AI Blog',
    tag: 'AI',
    date: 'March 12, 2025',
    title: 'Recurrent Neural Networks uncovered — The power of memory in deep learning',
    description:
      'Deep learning has transformed many fields. RNNs, useful for sequential data, are enhanced by LSTM and GRU. Transformers are now the standard for efficiently processing complex sequences.',
    href: '/blogs/rnn',
    color: 'neon-purple',
    readTime: '6 min read',
    content: `
    <h1>Deep Learning and Recurrent Neural Networks (RNNs)</h1>

    <p>Deep learning has made <strong>significant strides</strong> in recent years, transforming various fields, including <strong>computer vision</strong>, <strong>natural language processing (NLP)</strong>, and <strong>speech recognition</strong>. Among the most widely used neural network architectures, <strong>Convolutional Neural Networks (CNNs)</strong> have become the standard for image analysis due to their ability to detect spatial patterns at different scales. However, despite their effectiveness in the visual domain, CNNs show <strong>limitations</strong> when processing sequential data such as text, audio, or time series.</p>

    <h2>Why This Limitation?</h2>

    <p>Unlike images, where <strong>spatial relationships</strong> between pixels are paramount, sequential data requires understanding <strong>time</strong> and the <strong>order of elements</strong>. For example, in the sentence “I’m going to Paris tomorrow”, the word “Paris” gains meaning from “tomorrow”. A CNN, designed to analyze fixed, local patterns, cannot capture this essential <strong>temporal dependence</strong>.</p>

    <p>This is where <strong>Recurrent Neural Networks (RNNs)</strong> come in. Designed to process sequences of data, RNNs enable <strong>memory retention</strong> to influence future decisions by linking each sequence element to its predecessors. This makes them particularly suitable for tasks such as:</p>

    <ul>
        <li><strong>Natural Language Processing (NLP) 🗣️</strong>: machine translation, text generation.</li>
        <li><strong>Speech Recognition 🎙️</strong>: Siri, Google Assistant.</li>
        <li><strong>Stock Market Forecasting 📈</strong>: time series analysis.</li>
        <li><strong>Music Generation 🎵</strong>: creative models based on sequences.</li>
    </ul>

    <p>In this article, we will explore how RNNs work, analyze their limitations, and examine how <strong>LSTM</strong> and <strong>GRU</strong> networks have revolutionized their applications. Finally, we will discuss the evolution of recurrent networks toward <strong>Transformers</strong>, which now dominate artificial intelligence due to their ability to capture <strong>long-term dependencies</strong> efficiently.</p>

    <hr>

    <h2>Architecture of RNNs</h2>

    <p>Recurrent Neural Networks (<strong>RNNs</strong>) are a type of neural network designed to process <strong>sequential data</strong>. Unlike classical networks (such as feedforward networks) that analyze each input independently, RNNs are able to <strong>retain information from the past</strong> to influence future predictions through <strong>feedback loops</strong>.</p>

    <p>In some situations, such as predicting the next word in a sentence, it is essential to remember previous terms to generate a coherent response. Classical neural networks (without temporal memory) cannot handle these <strong>long-term dependencies</strong>, which motivated the creation of RNNs.</p>

    <h3>The Key Component: Hidden State 🧠</h3>

    <p>The <strong>hidden state</strong> acts as <strong>contextual memory</strong>, allowing RNNs to:</p>
    <ul>
        <li>Store relevant information from previous steps.</li>
        <li>Update memory at each time step.</li>
        <li>Influence future predictions through recurrent mechanisms.</li>
    </ul>

    <p>📌 <strong>Main features of RNNs</strong>:</p>
    <ul>
        <li>They maintain a <strong>temporal context</strong> by memorizing key information from sequences.</li>
        <li>They apply the <strong>same parameters (weights)</strong> to each element of the sequence, thus reducing the complexity of the model (<strong>parameter sharing</strong>).</li>
        <li>They allow processing of sequential data such as text, audio, or time series, by exploiting their <strong>temporal structure</strong>.</li>
    </ul>

    <img src="../public/img/CNNTrans/archiRnn.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:70%; height:auto;">

    <h3>Principal components of RNNs</h3>
    <img src="../public/img/CNNTrans/rnncom.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <h4>a) Input Layer</h4>
    <p>The input layer of an RNN processes each element of the sequence (like a word in a sentence) by transforming it into a <strong>dense vector representation</strong> via <strong>embedding vectors</strong>. These embeddings, often pre-trained (like Word2Vec, GloVe, or BERT), are crucial because they:</p>
    <ul>
        <li>Capture semantic relationships between words (e.g., similarity, antonymy).</li>
        <li>Reduce dimensionality compared to a classic one-hot encoding.</li>
    </ul>
    <p>For example, the words “cat” and “dog” (domestic animals) will have geometrically close vectors in the embedding space, unlike “cat” and “car”. This allows the RNN to:</p>
    <ul>
        <li>Understand <strong>analogies</strong> and <strong>lexical context</strong>.</li>
        <li>Generalize better on rare or unknown words.</li>
    </ul>
    <p>This step is fundamental to analyzing linguistic or temporal sequences in a coherent manner, because it encodes the information before it is processed by the recurrent layers (hidden states).</p>

    <h4>b) Hidden Layer</h4>
    <p>The hidden layer is the <strong>heart</strong> of an RNN, as it allows it to memorize the <strong>context</strong> and process data step by step. Unlike a classic neural network (like a feedforward), where inputs are processed in isolation, an RNN maintains a <strong>dynamic memory</strong> via its hidden state, a numerical vector that evolves at each step.</p>

    <h5>Two-Stage Operation</h5>
    <p>Each time step, the hidden layer receives:</p>
    <ul>
        <li>The current input (e.g., the word “beau” in the sentence “Il fait beau aujourd’hui”).</li>
        <li>The previous hidden state (a summary of the previous words, like “Il fait”).</li>
    </ul>
    <p>These two elements are combined via an <strong>activation function</strong> (e.g., tanh or ReLU) to generate:</p>
    <ul>
        <li>A <strong>new hidden state</strong> (updated with the current context).</li>
        <li>An <strong>output</strong> (optional, depending on the task).</li>
    </ul>

    <h5><strong>Concrete Example</strong></h5>
    <p>In the sentence “It’s nice today”:</p>
    <ul>
        <li>When the RNN processes “beau”, the hidden state already contains the information “It’s nice”.</li>
        <li>This allows us to understand that “beau” describes the weather, and not an object (“un beau tableau”).</li>
    </ul>

    <p><strong>What is this memory used for?</strong></p>
    <ul>
        <li><strong>Machine translation</strong>: Connecting words from a source language to a target (“Chat” → “Cat” taking into account gender).</li>
        <li><strong>Speech recognition</strong>: Deducing “ice cream” rather than “I scream” using the acoustic context.</li>
        <li><strong>Text generation</strong>: Producing “Il fait froid” after “Il neige en hiver, donc…”.</li>
    </ul>

    <p><strong>Why is it revolutionary?</strong></p>
    <ul>
        <li><strong>Parameter sharing</strong>: The same weights are used at each time step (saving computation).</li>
        <li><strong>Flexibility</strong>: Processes sequences of variable length (sentences, time series).</li>
    </ul>

    <h4>c) Activation function</h4>
    The <strong>activation function</strong> is a critical component of an RNN, as it introduces <strong>non-linearity</strong>, enabling the network to learn complex relationships between elements in a sequence. Without this transformation, the RNN would process information <strong>linearly</strong> (like a basic calculator), limiting its ability to capture complex dependencies such as irony, intensity, or grammatical nuances.</p>

    <h5>How Does the Activation Function Work in an RNN?</h5>
    <p>At each timestep, a hidden layer neuron receives two inputs:</p>
    <ul>
    <li><strong>Current input</strong> (e.g., a word in a sentence).</li>
    <li><strong>Previous hidden state</strong> (a numerical summary of past elements).</li>
    </ul>
    <p>These values are combined via a linear operation:</p>
    <p><code>h<sub>t</sub> = W<sub>x</sub> ⋅ x<sub>t</sub> + W<sub>h</sub> ⋅ h<sub>t-1</sub> + b</code></p>
    <p><strong>Issue</strong>: Without an <strong>activation function</strong>, this equation only allows proportional relationships (e.g., "twice as cold" = "twice as many clothes"), lacking contextual adaptation.</p>

    <p><strong>Solution</strong>: The <strong>activation function</strong> (e.g., <strong>tanh</strong> or <strong>ReLU</strong>) applies a <strong>non-linear transformation</strong>. This enables the <strong>RNN</strong> to:</p>
    <ul>
    <li>Capture <strong>conditional patterns</strong> (e.g., "very" amplifies "cold" but dampens "hot").</li>
    <li>Dynamically modulate word impact based on context.</li>
    </ul>

    <p><strong>Why is Non-Linearity Necessary?</strong></p>
    <p>Without it, each <strong>hidden state h(t)</strong> would be a linear combination of past inputs. The <strong>RNN</strong> would then act like a simple statistical model, unable to:</p>
    <ul>
    <li>Differentiate between "It's a bit cold" and "It's very cold".</li>
    <li>Distinguish "I love it!" (positive) from "I love it... not" (negative).</li>
    </ul>

    <h4>d) Output Layer</h4>
    The <strong>output layer</strong> converts the final <strong>hidden state</strong> into a usable prediction:</p>
    <p><strong>Prediction conversion</strong>:<br>
    Takes the last <strong>hidden state ht</strong> (full context) and converts it to:</p>
    <ul>
    <li>A word (e.g., next word in a translation).</li>
    <li>A class (e.g., "positive" or "negative" sentiment).</li>
    <li>A numerical value (e.g., predicted temperature).</li>
    </ul>
    <p><strong>Tailored activation functions</strong>:</p>
    <ul>
    <li><strong>Softmax</strong>: For probabilities (e.g., choosing among 10,000 possible words).</li>
    <li><strong>Sigmoid</strong>: For binary classification (e.g., spam vs. non-spam).</li>
    <li><strong>Linear</strong>: For regression (e.g., stock price prediction).</li>
    </ul>
    <p><strong>Application example</strong>:<br>
    Sentence: "It's raining, so I'll take my [...]"<br>
    - Hidden state: Encodes context "rain" + "take".<br>
    - Output layer → "umbrella" (using <strong>softmax</strong>).</p>


    <h2>Different RNN Architectures</h2>

    <h3>One-to-One Architecture</h3>
    <p>The <strong>One-to-One</strong> architecture is the simplest form, where a single input is mapped directly to a single output. This model lacks <strong>sequential processing</strong> or <strong>temporal dependencies</strong>, making it functionally similar to <strong>traditional neural networks</strong> (like a <strong>perceptron</strong>). It is often used as a baseline for comparing more complex <strong>RNN architectures</strong> (e.g., <strong>One-to-Many</strong> or <strong>Many-to-Many</strong>).</p>
    <p>In this model:</p>
    <ul>
    <li>A <strong>single input</strong> (x) is processed to generate a single output (y).</li>
    <li>The output is computed using a linear mathematical function: <code>y=wx+b</code></li>
    </ul>
    <p>where:<br>
    - w: <strong>Weight</strong> (determines the input's influence).<br>
    - b: <strong>Bias</strong> (offsets the prediction).</p>
    <img src="../public/img/CNNTrans/onetoone.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">


    <h3>One-to-Many Architecture</h3>
    <img src="../public/img/CNNTrans/onetomany.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">
    <p>The <strong>One-to-Many</strong> architecture is designed for scenarios where a single input generates a sequence of outputs. It excels in tasks requiring the transformation of a single data point into a structured, multi-step result.</p>

    <p><strong>How Does It Work?</strong></p>
    <ol>
    <li><strong>Single Input (x)</strong>: A single data point is fed into the network (e.g., an image, text prompt, or audio clip).</li>
    <li><strong>Sequential Outputs (y₀, y₁, ..., yₙ)</strong>: The network generates outputs step-by-step, building a sequence over time.</li>
    <li><strong>Internal Propagation</strong>: At each step, the network uses:
    <ul>
    <li>The previous <strong>hidden state</strong> (memory of past steps).</li>
    <li>The initial input or prior outputs to generate the next result.</li>
    </ul></li>
    </ol>
    <p>This <strong>recurrence</strong> allows the model to maintain <strong>contextual coherence</strong>.</p>

    <p><strong>Concrete Examples</strong></p>
    <ol>
    <li><strong>Text-to-Speech (TTS)</strong>:
    <ul>
    <li>Input: A text string (e.g., "Hello").</li>
    <li>Output: A time-series audio waveform pronouncing the phrase.</li>
    <li>Mechanism: The <strong>RNN</strong> converts text into phonemes, then synthesizes audio frames sequentially.</li>
    </ul></li>
    <li><strong>Music Generation</strong>:
    <ul>
    <li>Input: A seed note (e.g., C4) or genre tag (e.g., "jazz").</li>
    <li>Output: A melody composed of multiple notes (e.g., [C4, E4, G4, ...]).</li>
    <li>Mechanism: The <strong>RNN</strong> predicts note pitch, duration, and timing iteratively.</li>
    </ul></li>
    </ol>


    <h3>Many-to-One Architecture</h3>
    <img src="../public/img/CNNTrans/manytomany.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <p>In <strong>RNNs</strong>, the <strong>Many-to-One (N:1)</strong> architecture transforms a sequence of inputs into a single output. It is used to synthesize a sequence into a global value or category, such as:</p>
    <ul>
    <li><strong>Sentiment Analysis</strong>: Determining the emotion of a text ("Positive/Negative").</li>
    <li><strong>Sequence Classification</strong>: Identifying abnormal patterns in time-series data.</li>
    <li><strong>Time-Series Prediction</strong>: Estimating future values (e.g., stock prices).</li>
    </ul>

    <h4><strong>Many-to-One Architecture Schema</strong></h4>

    <p><strong>Example</strong>: Sentiment analysis of the sentence "I loved this movie!"</p>

    <p><strong>Sequential Inputs</strong>:</p>
    <img src="../public/img/CNNTrans/seqinput.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <p><strong>Detailed Propagation:</strong></p>
    <ol>
    <li><strong>Input 1 (X₁ = "I")</strong>:
    <ul>
    <li>Compute h₁:
      <p><code>h₁ = f(Wx·X₁ + Wh·h₀ + b)</code></p>
      (h₀ is initialized to zero or randomly)</li>
    </ul></li>
    <li><strong>Input 2 (X₂ = "loved")</strong>:
    <ul>
    <li>Compute h₂:
      <p><code>h₂ = f(Wx·X₂ + Wh·h₁ + b)</code></p></li>
    </ul></li>
    <li><strong>Input 5 (X₅ = "!")</strong>:
    <ul>
    <li>Compute h₅:
      <p><code>h₅ = f(Wx·X₅ + Wh·h₄ + b)</code></p>
      <strong>Output:</strong> <code>Y = softmax(Wy·h₅ + by)</code> → "Positive"</li>
    </ul></li>
    </ol>



    <h3>Many-to-Many Architecture</h3>
    <img src="../public/img/CNNTrans/manytomany1.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">
    <p>This architecture handles sequences where input and output lengths differ. It is split into two specialized components: an <strong>encoder</strong> and a <strong>decoder</strong>, enabling tasks like translation or text generation.</p>
    <p>This architecture handles input and output sequences of different lengths.</p>

    <p><strong>Examples</strong>:</p>
    <ul>
    <li>Translation ("Bonjour" → "Hello", "Comment ça va ?" → "How are you?").</li>
    <li>Speech synthesis (text → audio).</li>
    <li>Dialogue systems (question → response).</li>
    </ul>

    <h4><strong>Structure</strong></h4>

    <p><strong>Encoder</strong></p>
    <ul>
    <li><strong>Role</strong>: Transforms the input into a context (a vector of numbers).</li>
    <li><strong>Function</strong>:
    <ul>
    <li>Processes each input element (e.g., words in a sentence) one by one.</li>
    <li>Updates a <strong>hidden state</strong> (memory) at each step.</li>
    <li>The final <strong>hidden state</strong> (context) summarizes the entire input.</li>
    </ul></li>
    </ul>

    <p><strong>Decoder</strong></p>
    <ul>
    <li><strong>Role</strong>: Generates the output step by step, using the context.</li>
    <li><strong>Function</strong>:
    <ul>
    <li>Initializes its <strong>hidden state</strong> with the encoder's context.</li>
    <li>Generates an output element (e.g., a word) at each timestep.</li>
    <li>Uses its own previous output as input for the next step (<strong>autoregression</strong>).</li>
    </ul></li>
    </ul>


    <h2>Advantages and Disadvantages of RNNs</h2>

    <p><strong>Advantages of RNNs</strong>:</p>
    <ul>
    <li>Handle <strong>sequential data</strong> effectively, including text, speech, and time series.</li>
    <li>Process inputs of any length, unlike <strong>feedforward neural networks</strong>.</li>
    <li>Share <strong>weights</strong> across time steps, enhancing training efficiency.</li>
    </ul>

    <p><strong>Disadvantages of RNNs</strong>:</p>
    <ul>
    <li>Prone to <strong>vanishing and exploding gradient</strong> problems, hindering learning.</li>
    <li>Training can be challenging, especially for long sequences.</li>
    <li>Computationally slower than other <strong>neural network architectures</strong>.</li>
    </ul>

    <h2>What Are Different Variations of RNN?</h2>
    <p>Researchers have introduced new, advanced <strong>RNN architectures</strong> to overcome issues like <strong>vanishing and exploding gradients</strong> that hinder learning in long sequences.</p>
    <ul>
    <li><strong>Long Short-Term Memory (LSTM)</strong>: A popular choice for complex tasks. <strong>LSTM networks</strong> introduce <strong>gates</strong>, i.e., <strong>input gate</strong>, <strong>output gate</strong>, and <strong>forget gate</strong>, that control the flow of information within the network, allowing them to learn <strong>long-term dependencies</strong> more effectively than vanilla <strong>RNNs</strong>.</li>
    <li><strong>Gated Recurrent Unit (GRU)</strong>: Similar to <strong>LSTMs</strong>, <strong>GRUs</strong> use <strong>gates</strong> to manage information flow. However, they have a simpler architecture, making them faster to train while maintaining good performance. This makes them a good balance between complexity and efficiency.</li>
    <li><strong>Bidirectional RNN</strong>: This variation processes data in both <strong>forward</strong> and <strong>backward</strong> directions. This allows it to capture context from both sides of a sequence, which is useful for tasks like <strong>sentiment analysis</strong> where understanding the entire sentence is crucial.</li>
    <li><strong>Deep RNN</strong>: Stacking multiple <strong>RNN layers</strong> on top of each other, <strong>deep RNNs</strong> creates a more complex architecture. This allows them to capture intricate relationships within very long sequences of data. They are particularly useful for tasks where the order of elements spans long stretches.</li>
    </ul>

    <h2>Conclusion</h2>
    <p><strong>Recurrent Neural Networks</strong> have revolutionized <strong>deep learning</strong> by enabling models to process <strong>sequential data</strong> effectively. Despite their limitations, advancements like <strong>LSTM</strong>, <strong>GRU</strong>, and <strong>Bidirectional RNNs</strong> have significantly improved their performance. However, modern architectures like <strong>Transformers</strong> are now pushing the boundaries of <strong>sequence modeling</strong> even further, marking the next evolution in <strong>AI-driven tasks</strong>.</p>
  `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'webgl' }, { id: 'ar' }],
  },
  {
    id: 'serverless',
    slug: 'cnn2',
    image: '../img/CNNTrans/cnnspart1.jpg',
    alt: 'Serverless Blog',
    tag: 'AI',
    date: 'Mar 01, 2025',
    title: 'Deep Learning basics for video — Convolutional Neural Networks (CNNs) — Part 2',
    description:
      'This article introduces activation features such as Sigmoid, Tanh and ReLU, their advantages, and the problem of gradient vanishing slowing down learning. It also explains the role of backpropagation in adjusting weights, and describes the pooling and fully connected layers in CNNs for feature reduction and decision making.',
    href: '/blogs/cnn2',
    color: 'neon-purple',
    readTime: '5 min read',
    content: `
      <h2>Different types of activation functions</h2>

      <h3>Sigmoid Function</h3>
      <p>The sigmoid function is one of the most well-known activation functions in artificial intelligence and machine learning. It is defined by the following equation:</p>

      <img src="../public/img/CNNTrans/form1.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:30%; height:auto;">

      <p>This function transforms any input value into a number between <strong>0 and 1</strong>. Thanks to this property, it is often used to represent probabilities, making it an excellent choice for <strong>binary classification problems</strong>.</p>

      <h5>Limitations of the Sigmoid Function</h5>
      <p>Although the sigmoid function is useful, it has some drawbacks:</p>
      <ul>
          <li><strong>Saturation Effect</strong>: For very large or very small input values, the output is close to 0 or 1, making the model less sensitive to variations.</li>
          <li><strong>Vanishing Gradient Problem</strong>: In deep neural networks, the sigmoid function can cause slow learning, as gradients become too small.</li>
          <li><strong>Non-Zero-Centered Output</strong>: Unlike other functions such as tanh, the sigmoid function produces only positive values (between 0 and 1). This can slow down model convergence because it forces the network’s weight updates to be unbalanced, making optimization less efficient.</li>
      </ul>

      <h3>Tanh Function</h3>
      <p>The <strong>Tanh function</strong> (or Hyperbolic Tangent) is an <strong>activation function</strong> used in neural networks. It transforms an input value into a number between <strong>-1 and 1</strong>.</p>
      <p>Here is its mathematical formula:</p>
      <img src="../public/img/CNNTrans/form2.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:30%; height:auto;">

      <p>This means that, no matter the value of x, the output of the function will always be between <strong>-1 and 1</strong>.</p>

      <h5>Why is it useful?</h5>

      <p><strong>It centers values around zero</strong><br>
      Unlike the <strong>Sigmoid function</strong>, whose outputs are always <strong>positive (between 0 and 1)</strong>, the Tanh function produces both <strong>positive and negative</strong> values (between -1 and 1).</p>
      <ul>
          <li>This helps <strong>neural networks</strong> learn more efficiently because positive and negative values are better balanced.</li>
          <li>It reduces the risk of the model being biased toward only positive values.</li>
      </ul>

      <p><strong>It helps normalize data</strong><br>
      By keeping values within a <strong>symmetric range (-1 to 1)</strong>, Tanh allows for better <strong>weight adjustments</strong> in the neural network and accelerates learning.</p>

      <h3>ReLU function</h3>
      <p>The <strong>ReLU</strong> (Rectified Linear Unit) function is one of the most widely used activation function in deep learning. It is mathematically defined as:</p>

      <img src="../public/img/CNNTrans/form3.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:30%; height:auto;">

      <p>This means that:</p>
      <ul>
          <li>If <strong>x is positive</strong>, the function keeps it unchanged (i.e., f(x)=x).</li>
          <li>If <strong>x is negative or zero</strong>, the function outputs 0 (i.e., f(x)=0).</li>
      </ul>

      <h5>Why is it useful?</h5>

      <p><strong>Avoids the vanishing gradient problem</strong></p>
      <ul>
          <li>In deep neural networks, activation functions like <strong>Sigmoid or Tanh</strong> can cause very small gradients during backpropagation, slowing down learning (this is called the vanishing gradient problem).</li>
          <li>Since ReLU does not <strong>squash large values</strong>, it allows gradients to stay strong, making training much faster and more efficient.</li>
      </ul>

      <p><strong>Computationally efficient</strong></p>
      <ul>
          <li>ReLU is very simple to compute: it just requires checking whether the input is positive or not.</li>
          <li>This makes it much faster than other activation functions like Sigmoid or Tanh, which involve exponentials.</li>
      </ul>


      <h3>What is vanishing gradient?</h3>
      <p>Before talking about the Vanishing gradient, let’s understand neural network weights and how to adjust them with backpropagation.</p>
      <p>When training a neural network, such as a Convolutional Neural Network (CNN), the aim is to reduce the error between what the model predicts and what the actual data shows. This is achieved by adjusting the network’s weights. But what are these weights?</p>

      <h5>What are neural network weights?</h5>
      A neural network is made up of layers, each containing neurons. Each connection between these neurons is associated with a weight, which is simply a number. This weight determines the importance of a connection: a high weight means that the connection has a great influence, while a low weight means that the connection has little impact on the calculation.</p>
      <p>During training, the network learns by modifying these weights to improve predictions. It’s a bit like adjusting the knobs on a radio to get clear sound: you adjust the weights to <strong>get the right prediction</strong>.</p>

      <h5>How Are Weights Adjusted?</h5>
      <p>To adjust the weights in a neural network, we use a method called <strong>backpropagation of the gradient</strong> (or simply Backpropagation). This process consists of three main steps:</p>

      <p>1️⃣ <strong>Calculating the Error (Loss Function)</strong><br>
      The first step is to <strong>measure the error</strong> between the model’s prediction and the actual value. This is done using a mathematical function called the Loss Function.</p>

      <p>📌 <strong>Examples of Loss Functions:</strong></p>
      <ul>
          <li>For a <strong>classification problem</strong> (e.g., predicting “dog” or “cat”), we often use <strong>Cross-Entropy Loss</strong>.</li>
          <li>For <strong>predicting continuous values</strong> (e.g., estimating a price), we use <strong>Mean Squared Error (MSE)</strong>.</li>
      </ul>
      <p>📌 <strong>Mathematical Example of Cross-Entropy Loss:</strong></p>
      <img src="../public/img/CNNTrans/form3.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:30%; height:auto;">
      <p>where:</p>
      <ul>
          <li><code>y<sub>i</sub></code> is the true class label (e.g., “dog” or “cat”).</li>
          <li><code>ŷ<sub>i</sub></code> the predicted probability assigned to that class.</li>
      </ul>
      <p>👉 <strong>This formula produces a value that represents the total error.</strong><br>
      <strong>The goal is to minimize this error as much as possible!</strong></p>

      <p>2️⃣ <strong>Computing the Gradient: Finding the Right Direction</strong><br>
      Once the error is calculated, the model needs to adjust its weights to reduce it. <strong>But how do we know in which direction to change the weights?</strong> This is where the concept of gradient comes into play.</p>

      <p><strong>What is a Gradient?</strong><br>
      A <strong>gradient</strong> is a mathematical measure that tells us <strong>how much and in which direction a value should change</strong>. In our case, it measures <strong>how the loss function changes with respect to each weight in the network</strong>.</p>
      <p>📌 <strong>Mathematically, we compute the partial derivative of the loss function with respect to each weight:</strong></p> 
      <img src="../public/img/CNNTrans/form4.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:30%; height:auto;">
      

      <p>📌 <strong>How do we interpret this?</strong></p>
      <ul>
          <li>If the <strong>gradient is positive</strong>, we need to decrease the weight to reduce the error.</li>
          <li>If the <strong>gradient is negative</strong>, we need to increase the weight to reduce the error.</li>
          <li>If the <strong>gradient is close to zero</strong>, the weight stops changing significantly.</li>
      </ul>
      <p>⚠️ <strong>This is where the Vanishing Gradient Problem occurs!</strong><br>
      When gradients become extremely small, the network struggles to update the weights in early layers, making learning very slow or even impossible</p>

      <p>3️⃣ <strong>What is Backpropagation?</strong><br>
      Backpropagation is a method used in neural networks to correct their errors. The main idea is to <strong>adjust the weights</strong> of the network (the “knobs” that control the strength of connections between neurons) so that the network makes better predictions next time.</p>
      <ul>
      <li><strong>The network makes an error</strong>:<br>When we give an input to the network (e.g., a picture of a cat), it produces an output (e.g., “dog” instead of “cat”). We compare this output to the correct answer to measure the <strong>error</strong>.</li>
      <li><strong>Calculating how much each connection (weight) contributed to the error</strong>:<br>
      We start by looking at the <strong>last layer</strong> (the one that gives the final answer) and compute how much each weight influenced the error. Then, we move backward layer by layer, adjusting the weights at each step.</li>
      <li><strong>Using the gradient and the chain rule</strong>:<br>
      When we adjust the weights, we need to understand how each weight influences the model’s error. To do this, we use the <strong>gradient</strong>, which tells us how much the error (or loss) changes with respect to each weight. The chain rule is a mathematical concept that allows us to compute this gradient efficiently, even when there are multiple layers in the network.</li>
      </ul>
      <p>Imagine you need to understand how an action in a previous layer affected the final error. To do that, you have to trace how this action affects the next layer, and so on, until you reach the output. It’s like a <strong>domino effect</strong>, where each domino influences the one that follows it.</p>
      <p>Here’s how we proceed:</p>
      <ul>
      <li>The <strong>final error (L)</strong> depends on the output of the network (O).</li>
      <li>The <strong>output (O)</strong> depends on the activation (H) of the previous layer.</li>
      <li>The <strong>activation (H)</strong> depends on the weights (W) of the layer.</li>
      </ul>
      <p>By using the chain rule, we can combine these effects to calculate the impact of each weight on the final error.</p>
      <p><strong>Simple Chain Rule Formula:</strong></p>

      <img src="../public/img/CNNTrans/form5.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:30%; height:auto;">

      <p>This means:</p>
      <ul>
      <li><code>∂L/∂O​</code>: Measures how much the error changes with respect to the output.</li>
      <li><code>∂O/∂H​</code>: Measures how much the output changes with respect to the activation of the previous layer.</li>
      <li><code>∂H/∂W<sub>1</sub>​</code>: Measures how much the activation changes with respect to the weight of the layer.</li>
      </ul>
      <p>This <strong>cascade multiplication</strong> of gradients through the layers allows the network to understand the effect of each weight on the final error and adjust the weights accordingly. If a weight had a significant impact on the error, its gradient will be larger, and it will be adjusted more. If it had little impact, its gradient will be smaller, and it will be adjusted less.</p>

      <ul>
      <li><strong>Updating the weights:</strong><br>
      Once we have computed the gradients, we update the weights to reduce the error. For example:
      <ul>
      <li>If the gradient for a weight is <strong>positive</strong>, we decrease the weight.</li>
      <li>If the gradient is <strong>negative</strong>, we increase the weight.</li>
      <li>If the gradient is <strong>0</strong>, the weight stops changing significantly.</li>
      </ul></li>
      </ul>
      <p>This process is repeated over many iterations (or epochs), with the weights being adjusted gradually, improving the network’s ability to make accurate predictions!</p>


      <p>The <strong>Vanishing Gradient</strong> Problem occurs when gradients become very small (close to zero) in the early layers of the network. This means that the weights in these layers stop updating properly, causing the first layers to stop learning effectively.</p>

      <p><strong>Why does this happen?</strong><br>
      Some activation functions, like Sigmoid or Tanh, have very small derivatives when their inputs are either very large or very small. This weakens the gradient as it propagates backward through the network.</p>

      <p><strong>Example with the Sigmoid Function</strong><br>
      The Sigmoid function is defined as:</p>
      <img src="../public/img/CNNTrans/form6.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:25%; height:auto;">
      <p>Its derivative is:</p>
      <img src="../public/img/CNNTrans/form7.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:30%; height:auto;">
      <p>👉 When x is very large or very small, f(x) becomes close to 1 or 0, and its derivative approaches zero.</p>
      <p>In a <strong>deep network</strong>, the gradient is computed layer by layer, multiplying these derivatives at each step.</p>
      <ul>
      <li>If the <strong>derivative is always small</strong> (&lt;1), then gradients shrink with each multiplication.</li>
      <li>By the time they reach the <strong>earlier layers</strong>, they become almost zero → the first layers stop learning.</li>
      </ul>

      <p><strong>Why is this a problem?</strong></p>
      <ul>
      <li>🔴 The <strong>first layers stop learning</strong> → They remain unchanged and fail to capture basic features like edges and shapes.</li>
      <li>🔴 <strong>Learning becomes very slow</strong> → Only the last layers (closer to the output) receive meaningful updates.</li>
      <li>🔴 The <strong>network becomes ineffective</strong> → It struggles to learn complex patterns from the data.</li>
      </ul>

      <p><strong>How to Avoid the Vanishing Gradient?</strong></p>
      <ul>
      <li>✅ <strong>Use ReLU Instead of Sigmoid or Tanh</strong><br>
      👉 The <strong>ReLU function</strong> is defined as:<br>
      <strong>f(x)=max⁡(0,x)</strong><br>
      👉 Its derivative is <strong>1 for positive values</strong>, meaning it does not shrink the gradient, preventing it from vanishing.</li>
      <li>✅ <strong>Use Batch Normalization</strong><br>
      👉 This technique normalizes the input values of each layer to prevent extreme values and stabilize learning, ensuring smoother training.</li>
      <li>✅ <strong>Use Proper Weight Initialization (Xavier, He)</strong><br>
      👉 Proper weight initialization prevents values from being too large or too small at the start, reducing the risk of activation saturation and gradient shrinkage.</li>
      <li>✅ <strong>Use Advanced Optimizers (Adam, RMSprop)</strong><br>
      👉 These optimization algorithms automatically adjust weight updates to prevent the gradient from becoming too small, ensuring efficient learning.</li>
      </ul>

      <p>👉 <strong>Activation layers</strong> allow the network to model complex patterns instead of being limited to simple linear relationships. Once these complex features are extracted, pooling layers are used to reduce the data size while retaining the most important information. They act as a summary of the previous layers, making the network more efficient and less computationally demanding.</p>

      <h3>Pooling Layers</h3>
      <p><strong>Pooling</strong> is an operation that reduces the size of feature maps while preserving key information. It works by grouping neighboring pixels and summarizing their content into a single value. This helps:</p>
      <ul>
      <li>✅ <strong>Reduce data dimensions</strong>, speeding up computations.</li>
      <li>✅ <strong>Preserve essential features</strong> (such as edges and contours).</li>
      <li>✅ <strong>Increase model robustness</strong> against minor variations in an image (noise, shifts, and transformations).</li>
      </ul>

      <img src="../public/img/CNNTrans/matrixConv.jpg" alt="architecture" style="display: block; margin: 0 auto; max-width:30%; height:auto;">

      <h5>How Does Pooling Work?</h5>
      <p>Pooling is a crucial step in convolutional neural networks (CNNs) used to reduce the size of data (images or feature maps) while retaining the most important information. Here’s how it works:</p>
      <p>We define a <strong>small square region</strong> of the image (usually 2×2 or 3×3 pixels) and apply a pooling function to this region to compute a single value that represents it.</p>
      <p>There are various types of pooling, each with unique advantages and use cases:</p>

      <p>1️⃣ <strong>Max Pooling (The Most Common Type)</strong></p>
      <ul>
      <li><strong>How Does It Work?</strong><br>
      Max pooling selects the maximum value in the defined region.</li>
      <li><strong>Why Use It?</strong><br>
      ✅ It captures the <strong>most prominent features</strong> of the image, such as <strong>edges and contours</strong>.<br>
      ✅ It effectively preserves visually important details.<br>
      ❌ However, it can be <strong>sensitive to noise</strong>: if an outlier pixel has a very high value, it will be retained.</li>
      </ul>
      <p>📌 <strong>When to Use Max Pooling?</strong><br>
      ✅ When you want to keep the <strong>dominant structures of an image</strong>.<br>
      ✅ Ideal for <strong>convolutional neural networks</strong> used in computer vision tasks (e.g., image classification, object detection).</p>

      <p>2️⃣ <strong>Average Pooling</strong></p>
      <ul>
      <li><strong>How Does It Work?</strong><br>
      Average pooling calculates the mean of the values in the defined region.</li>
      <li><strong>Why Use It?</strong><br>
      ✅ It smooths out variations and reduces the impact of extreme values.<br>
      ✅ It is <strong>less sensitive to noise</strong> than max pooling.<br>
      ❌ However, it may dilute important features like edges and contrasts.</li>
      </ul>
      <p>📌 <strong>When to Use Average Pooling?</strong><br>
      ✅ When you want to reduce the size of the image without losing too much information.<br>
      ✅ Suitable for tasks requiring smoother feature maps, such as speech recognition.</p>

      <p>3️⃣ <strong>L2 Norm Pooling</strong></p>
      <p><strong>How Does It Work?</strong><br>
      L2 Norm Pooling computes the L2 norm of the values in the region, which is the square root of the sum of squares of the values.</p>
      <p><strong>Why Use It?</strong><br>
      ✅ It provides a measure of the <strong>overall intensity</strong> of the region.<br>
      ❌ It’s less commonly used than Max or Average Pooling but can be useful for specific tasks.</p>
      <p>📌 <strong>When to Use L2 Norm Pooling?</strong><br>
      ✅ When you need a robust measure of pixel intensity, such as in industrial vision applications.</p>

      <p>4️⃣ <strong>Weighted Average Pooling</strong></p>
      <p><strong>How Does It Work?</strong><br>
      Weighted Average Pooling computes a weighted mean, assigning more importance to central pixels in the region.</p>
      <p><strong>Why Use It?</strong><br>
      ✅ It is well-suited to images where the center contains the most critical information (e.g., in facial recognition).<br>
      ❌ It is more computationally complex than other methods.</p>
      <p>📌<strong>When to Use Weighted Average Pooling?</strong><br>
      ✅ When preserving central details of an image is important, such as in medical imaging (e.g., analyzing MRI scans).</p>

      <p><strong>How Pooling Fits into CNNs</strong><br>
      Pooling is typically applied after convolution and activation layers. Here’s how they work together:</p>
      <ul>
      <li><strong>Convolution Layers</strong> extract features from the image by applying filters.</li>
      <li><strong>Activation Layers</strong> (like ReLU) highlight important features by introducing non-linearity.</li>
      <li><strong>Pooling Layers</strong> simplify the output by summarizing the most important information, reducing the data size and computational complexity.</li>
      </ul>

      <h3>Flatten Layers</h3>
      <p>1️⃣ <strong>What is Flattening?</strong><br>
      Flattening is a simple but essential step in a Convolutional Neural Network (CNN).<br>
      It is the process of <strong>transforming a 2D feature map</strong> into a 1D vector.</p>
      <p>👉 <strong>Before Flattening</strong>: The data is in a 2D shape (Height × Width × Depth).<br>
      👉 <strong>After Flattening</strong>: The data becomes a long 1D vector, which can be fed into a traditional dense neural network layer.</p>

      <p>2️⃣ <strong>Why is Flattening Necessary?</strong><br>
      Convolutional and pooling layers process images as matrices (2D or 3D). However, fully connected (dense) layers in a neural network expect a 1D vector as input.</p>
      <p>✅ <strong>Flattening converts the feature map into a format that the fully connected layers can process.</strong></p>


      <h3>Fully Connected Layers (FC)</h3>
      <p><strong>Fully connected layers</strong>, also called dense layers, are a fundamental component of neural networks, especially in Convolutional Neural Networks (CNNs) and Deep Neural Networks (DNNs).</p>
      <p>In a fully connected layer, every neuron is connected to all the neurons in the previous layer. This is in contrast to convolutional layers, where the connections are local and limited to small regions of the input (e.g., receptive fields).</p>
      <p>Here’s how it works:</p>
      <ul>
      <li>Each neuron in the fully connected layer receives inputs from all the outputs of the previous layers.</li>
      <li>These inputs are multiplied by weights, and a bias is added.</li>
      <li>A non-linear activation function (like ReLU or Sigmoid) is then applied to produce the neuron’s output.</li>
      </ul>
      <p>The <strong>fully connected layers</strong> combine all the extracted features from convolutional and pooling layers to produce the final decision of the network, whether it’s for classification (e.g., predicting a class label) or regression (e.g., outputting a continuous value).</p>

      <h5>Why are Fully Connected Layers Important?</h5>
      Fully connected layers play a key role in CNNs by:</p>
      <ul>
      <li>Taking the <strong>features extracted</strong> by convolutional and pooling layers.</li>
      <li><strong>Combining them</strong> to produce a final output (e.g., probabilities of classes in classification tasks).</li>
      <li>Acting as the “brain” of the network, where all the information is synthesized for the final prediction.</li>
      </ul>
      <p>Typically, fully connected layers appear at the end of a CNN and are used for tasks like classification or regression based on the processed information from earlier layers.</p>

      <p>Now that we’ve covered the foundational components of CNNs, let’s move on to discuss <strong>Recurrent Neural Networks (RNNs)</strong> 👉</p>
    `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'ai' }, { id: 'microservices' }],
  },
  {
    id: 'ar',
    slug: 'cnn1',
    image: '../img/CNNTrans/cnnspart1.jpg',
    alt: 'AR Commerce Blog',
    tag: 'AI',
    date: 'Feb 20, 2025',
    title: 'Deep Learning basics for video — Convolutional Neural Networks (CNNs) — Part 1',
    description:
      'This article introduces the basics of Convolutional Neural Networks (CNNs), explaining how they use filters to detect image features like edges and shapes. It highlights the role of activation layers in enabling CNNs to learn complex patterns for tasks such as object recognition.',
    href: '/blogs/cnn1',
    color: 'neon-cyan',
    readTime: '6 min read',
    content: `
    <h2>Deep Learning basics for video — Convolutional Neural Networks (CNNs) — Part 1</h2>
    <p>With the rise of deep learning, Convolutional Neural Networks (CNNs) have become a key technology, transforming image and video processing. These specialized neural networks are designed to replicate the functioning of the human visual cortex, allowing machines to analyze images by automatically extracting relevant visual features.</p>

    <p>Since their introduction in 1989 for handwritten digit recognition, CNNs have evolved significantly, becoming essential in various fields that require image analysis and complex data processing. Their effectiveness comes from their ability to detect visual patterns with remarkable accuracy, often outperforming traditional computer vision techniques.</p>

    <p>Today, CNNs play a vital role in multiple applications, including:</p>

    <ul>
      <li><strong>Computer vision</strong>, such as image classification, facial recognition, object detection, and segmentation.</li>
      <li><strong>Natural Language Processing (NLP)</strong>, where convolutions are applied to matrix representations of words for text analysis.</li>
      <li><strong>Recommendation systems</strong>, by interpreting user preferences based on images and videos.</li>
    </ul>

    <p>In this section, we will examine the structure of CNNs and their fundamental components.</p>

    <h2>Structure and Components of CNNs</h2>

    <p>A <strong>Convolutional Neural Network</strong> (CNN) is a specialized neural architecture designed to efficiently process visual data. Its operation relies on a series of layers that extract, transform, and interpret the features of an image to deduce a classification or prediction.</p>

    <p>Thanks to this hierarchical architecture, a CNN can capture simple patterns (edges, textures) in the early layers and more complex structures (shapes, objects) in the deeper layers.</p>

    <p>A CNN consists of several types of layers, each playing a specific role in image analysis:</p>

    <ul>
      <li><strong>Convolution</strong> — The core operation of the CNN, it applies filters to extract visual features (edges, patterns, textures).</li>
      <li><strong>Activation</strong> — After convolution, an activation function, such as ReLU (Rectified Linear Unit), is applied to introduce non-linearity, enabling the network to learn complex relationships.</li>
      <li><strong>Pooling</strong> — A dimensionality reduction technique that preserves essential information while reducing the complexity of the model.</li>
      <li><strong>Fully Connected Layers</strong> — These layers transform the extracted features into final decisions, such as image classification.</li>
    </ul>

    <h2>Convolution: Extraction of Visual Characteristics</h2>

    <p>An image can be represented as a matrix of pixels, where each pixel contains a light intensity (for a grayscale image) or several values (Red, Green, Blue — RGB) for a color image.</p>

    <p>But how can a machine identify shapes, textures or objects from this raw data? This is where convolution comes in! It is a mathematical operation that allows to extract characteristic patterns from an image and reveal its essential structures. This operation applies filters to an image to detect specific patterns, such as:</p>

    <ul>
      <li>🖼️ Edges</li>
      <li>🎨 Textures</li>
      <li>🔳 Shapes and geometric structures</li>
    </ul>

    <p>Each filter acts like a lens, highlighting certain aspects of the image, making it easier for machines to automatically recognize visual elements.</p>

    <h3>🔍 A Simple Analogy: The Magnifying Glass</h3>

    <p>Imagine looking at an image through a <strong>magnifying glass</strong>. As you move it across different parts of the image, you can observe specific details more clearly, such as the edges of an object or a unique texture. <strong>Convolution</strong> does the same thing with a <strong>filter</strong>, but in a mathematical and systematic way.</p>

    <h3>Steps of Convolution</h3>

    <h4>1️⃣ Choosing the Filter (Convolution Kernel)</h4>

    <p>A <strong>filter</strong> is a <strong>small matrix of numbers</strong> (often 3×3 or 5×5) that interacts with an image to highlight specific features. Different types of filters serve distinct purposes:</p>

    <ul>
      <li><strong>Edge detection filters</strong> (Sobel, Prewitt, Laplacian): Highlight <strong>sudden changes in brightness</strong>, making the edges of objects more visible.</li>
    </ul>
    <img src="../public/img/CNNTrans/grayeffect.jpg" alt="Edge detection with Sobel filter" />

    <ul>
      <li><strong>Blurring filter (Gaussian Blur)</strong>: Applies a weighted average to smooth an image and reduce noise.</li>
    </ul>
    <img src="../public/img/CNNTrans/blurring.jpg" alt="Gaussian blur application" />

    <ul>
      <li><strong>Sharpening filter</strong>: Enhances edges and improves image sharpness.</li>
    </ul>
    <img src="../public/img/CNNTrans/sharpening.jpg" alt="Sharpen filter application" />

    <h4>2️⃣ Applying the Filter to the Image (Convolution Operation)</h4>

    <img src="../public/img/CNNTrans/filterapply.gif" alt="Filter application example" />

    <ul>
      <li>The <strong>filter</strong> is applied to a specific region of the image.</li>
      <li>A <strong>pointwise multiplication</strong> is performed between the filter values and the corresponding pixel values.</li>
      <li>The sum of these results becomes the new pixel in the transformed image.</li>
      <li>The filter is then <strong>shifted</strong> to a new region, and the operation is repeated until the entire image has been processed.</li>
    </ul>

    <p>Each resulting value depends on the applied filter. An edge detection filter will highlight the edges, while a blurring filter will smooth out fine details. The image obtained after convolution is called the <strong>Feature Map</strong>. It highlights essential information while eliminating irrelevant details.</p>

    <h3>Mathematical Formula for the Convolution</h3>

    <p>The convolution on an image <em>I</em> with a filter <em>K</em> is expressed as:</p>

    <div class="equation">
      <img src="../public/img/CNNTrans/form8.jpg" alt="Filter application example" />
    </div>

    <ul>
      <li><strong>S(i,j)</strong>: Value of the resulting pixel after convolution at position (i,j).</li>
      <li><strong>I(i+m,j+n)</strong>: Value of the original image pixel, shifted by (m,n).</li>
      <li><strong>K(m,n)</strong>: Value of the kernel (filter) applied to the image region.</li>
      <li><strong>k</strong>: Half the size of the filter; if the filter is of size N x N, then k = (N-1)/2.</li>
      <li><strong>Σ</strong>: Summation operation covering the entire area affected by the filter.</li>
    </ul>

    <h2>Activation Layers</h2>

    <p>After the convolution stage, where filters are applied to extract specific features from the image, the network generates a feature map. This map highlights relevant elements while eliminating unnecessary details. However, convolution alone has limitations. It can detect simple features, like edges, textures, or patterns, but it cannot model complex relationships or understand non-linearities in the data.</p>

    <p>To address this limitation, an <strong>activation function</strong> is applied to each value in the feature map. This function introduces <strong>non-linearity</strong> into the model, which is essential for enabling the network to learn complex relationships and perform tasks like classification or object detection.</p>

    <h3>Why Use Activation Layers After Convolution?</h3>

    <p>Convolution is a powerful operation for feature extraction, but it is inherently linear. A linear operation satisfies two properties:</p>

    <ul>
      <li><strong>Proportionality</strong>: If the input is multiplied by a constant, the output is also multiplied by the same constant.</li>
      <li><strong>Additivity</strong>: The sum of two inputs corresponds to the sum of the two outputs.</li>
    </ul>

    <p>While convolution can detect simple relationships, such as brightness differences between pixels, it cannot handle complex or non-linear relationships, which are necessary for tasks like recognizing objects in varying conditions.</p>

    <h3>Limitations of Linearity in Convolution</h3>

    <p>Consider the example of recognizing a cat in an image. Convolution alone can detect basic features, such as edges (like the cat’s ears) or repetitive patterns (such as fur texture). However, these features are insufficient for recognizing a cat, especially in more challenging scenarios like:</p>

    <ul>
      <li>A cat with a bent ear</li>
      <li>A partially hidden cat</li>
      <li>A cat seen from an unusual angle</li>
    </ul>

    <p>In these cases, convolution struggles to combine simple features and understand complex relationships, such as “these contours form a cat’s face.” This is where activation functions come in.</p>

    <h3>Role of Activation Functions</h3>

    <p>Activation functions introduce non-linearity into the network, enabling it to model complex relationships. This is critical for tasks that involve more than simple linear transformations, such as recognizing objects that change in size, orientation, brightness, or position.</p>

    <p>In a convolutional network, each layer progressively learns to detect more abstract features:</p>

    <ul>
      <li>The <strong>first layers</strong> identify simple patterns (e.g., edges, textures).</li>
      <li><strong>Intermediate layers</strong> combine these patterns to detect shapes (e.g., ears, eyes).</li>
      <li><strong>Deeper layers</strong> integrate these shapes to recognize objects (e.g., a cat’s face).</li>
    </ul>

    <p>Without non-linearity, this hierarchy of abstraction would not be possible, as each layer would merely perform a linear combination of data from the previous layer.</p>

    <p><em>Now, let’s talk about different types of activation functions in part 2 👉</em></p>

  `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'webgl' }, { id: '3d' }],
  },
  {
    id: 'microservices',
    slug: 'risecnn',
    image: '../img/CNNTrans/cnn.jpeg',
    alt: 'Microservices Blog',
    tag: 'AI',
    date: 'Feb 18, 2025',
    title: 'The Rise of Smart Video Processing - From CNNs to Transformers',
    description:
      'This article explores the rise of video content and how video understanding has advanced—from analyzing single frames to using deep learning and Transformers to capture spatial and temporal information. These innovations now power smarter, faster analysis in areas like security, healthcare, and autonomous driving, with future sections covering the techniques behind them.',
    href: '/blogs/risecnn',
    color: 'neon-purple',
    readTime: '8 min read',
    content: `
    <h2>The Rise of Smart Video Processing - From CNNs to Transformers</h2>  
    <p>Today, video has become an essential means of communication. According to Sandvine's Global Internet Phenomena Report 2023, video content accounts for more than 65% of global Internet traffic, with an estimated annual growth of 24%. This spectacular evolution has led to the rise of numerous applications, ranging from online education to telemedicine, as well as digital advertising, virtual reality (VR), and many other fields.</p>

    <p>Despite its growing importance, video processing and manipulation are still absent from many school curricula. That's why this blog invites you to explore this fascinating universe, covering topics such as video understanding, captioning, localization, and much more.</p>

    <p>Whether you are a curious beginner or a passionate learner seeking improvement, this blog will guide you through the essential techniques to better understand and master video.</p>

    <p>📌 <strong>Ready to dive into the captivating world of video processing? Then let's begin!</strong> 🚀</p>

    <h2>The Evolution of Video Understanding: From Frames to Intelligence</h2>

    <p>Understanding a video is much more than just watching a sequence of moving images. It's a fascinating technical challenge where artificial intelligence plays a key role. Historically, the first approaches aimed to break down videos by analyzing each frame individually. These methods attempted to identify important visual elements, such as edges or shapes, to convert them into usable data.</p>

    <p>But here's the issue: these techniques had limitations. They required complex manual adjustments and struggled with unpredictable situations, such as fast movements, lighting changes, or hidden objects. Fortunately, the arrival of Deep Learning changed everything. These revolutionary algorithms have automated video analysis, making it faster, more accurate, and, above all, more adaptable.</p>

    <h3>The First Step: 2014–2016</h3>

    <p>This is where it all began. Researchers introduced the first neural networks for video analysis, laying the foundation for everything that followed. While these early models had limitations, they paved the way for major breakthroughs.</p>

    <ul>
      <li><strong>Two-Stream CNNs</strong>: A simple yet powerful idea — one network processed images (to capture visual details), while another focused on motion (to analyze actions).</li>
      <li><strong>3D CNNs</strong>: A game-changer that enabled the simultaneous analysis of both spatial and temporal dimensions.</li>
      <li><strong>CNN-LSTMs</strong>: By combining Convolutional Neural Networks (CNNs) with Long Short-Term Memory (LSTM) networks, these models brought a deeper understanding of sequences, making video analysis far more intelligent.</li>
    </ul>

    <p>These pioneering approaches were promising but still lacked the speed and accuracy required for real-world applications.</p>

    <h3>The Era of Improvements: 2017–2019</h3>

    <p>In the following years, video analysis took a huge leap forward. The focus was no longer just on identifying objects or actions in individual frames — it became about merging spatial and temporal information to interpret videos in a more holistic way.</p>

    <ul>
      <li><strong>ActionVLAD</strong> introduced a groundbreaking approach to combine spatial and temporal data, ensuring more coherent and meaningful video analysis.</li>
      <li><strong>R(2+1)D</strong> built on previous methods by breaking down 3D convolutions into separate spatial and temporal steps, improving action recognition.</li>
      <li><strong>TSM</strong> (Temporal Shift Module) and <strong>SlowFast Networks</strong> revolutionized video analysis:
        <ul>
          <li>TSM enabled faster processing without sacrificing accuracy.</li>
          <li>SlowFast captured both fast-moving details and long-term context, significantly enhancing performance.</li>
        </ul>
      </li>
    </ul>

    <p>At the same time, the introduction of large-scale datasets like Kinetics-400 provided the perfect training environment. These datasets contained diverse and realistic videos, allowing models to generalize better to real-world scenarios.</p>

    <p>💡 In short, video AI wasn't just getting smarter — it was becoming faster, more adaptable, and more powerful than ever before!</p>

    <h3>The Rise of Transformers: 2020 and Beyond</h3>

    <p>We have now entered an era where AI-powered video understanding has reached unprecedented levels. Thanks to cutting-edge architectures, models can analyze videos more accurately, efficiently, and intelligently than ever before.</p>

    <ul>
      <li><strong>Video Transformers (TimeSformer)</strong>: Traditional CNN-based approaches had limitations in capturing long-range dependencies across frames. TimeSformer introduced self-attention mechanisms, allowing models to analyze global relationships throughout a video rather than just local patterns. This resulted in higher precision and better contextual understanding.</li>
      <li><strong>Hierarchical Models (MViT & VideoSwin)</strong>:
        <ul>
          <li><strong>MViT</strong> (Multiscale Vision Transformer) refined video analysis by progressively learning representations at different scales, making it more adaptable to complex scenes.</li>
          <li><strong>VideoSwin</strong> extended the Swin Transformer concept to videos, using shifted windows for efficient yet powerful video processing.</li>
        </ul>
      </li>
      <li><strong>Lightweight Models (X3D)</strong>: While accuracy is essential, real-world applications also demand speed and efficiency. X3D (Expandable 3D CNN) optimized both accuracy and computational cost, making high-quality video analysis accessible on a large scale.</li>
    </ul>

    <h3>Why Are These Advances Crucial?</h3>

    <p>The evolution of video analysis technologies represents much more than just technical progress. It reflects a deep transformation in how machines interact with the visual world. Moving from traditional, manual, and limited methods to intelligent algorithms capable of interpreting videos with unprecedented accuracy and speed has opened the door to revolutionary applications.</p>

    <p>These advances are crucial in many fields:</p>
    <ul>
      <li><strong>Security and surveillance</strong>: Identifying suspicious behaviors in real time</li>
      <li><strong>Autonomous driving</strong>: Understanding the road environment to prevent accidents</li>
      <li><strong>Media and entertainment</strong>: Offering more relevant recommendations and immersive experiences</li>
      <li><strong>Healthcare</strong>: Analyzing medical videos to detect anomalies with precision</li>
    </ul>

    <p>Each step of this evolution has helped solve major technical challenges while making these tools more accessible and versatile. These technologies no longer just "see"; they understand, interpret, and react. This is a breakthrough that redefines the standards of what artificial intelligence can achieve, paving the way for future innovations.</p>

    <h3>What's Next: Exploring Techniques and Technologies</h3>

    <p>In our next article, we will dive deeper into the various techniques and technologies that make video analysis possible. We will explore modern approaches, from convolutional neural networks (CNNs) to the latest video transformers.</p>

    <p>We will also detail fundamental concepts such as spatiotemporal analysis, self-attention mechanisms, and lightweight architectures, illustrating their role in real-world applications.</p>

    <p><strong>Stay with us to discover how these technological solutions are transforming not only how videos are understood but also the industries in which they are used!</strong></p>

    `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'serverless' }, { id: 'ai' }],
  },
  {
    id: 'microservices',
    slug: 'hfl',
    image: '../img/Federated/HFL_cover.png',
    alt: 'Microservices Blog',
    tag: 'AI',
    date: 'March 03, 2024',
    title: 'Horizontal Federated Learning',
    description:
      'This article highlights key findings from Horizontal Federated Learning (HFL). It demonstrates that HFL is beneficial for preserving data privacy and is feasible in practice. Furthermore, the article shows that HFL can achieve good results with the appropriate hyperparameters and tools.',
    href: '/blogs/hfl',
    color: 'neon-purple',
    readTime: '8 min read',
    content: `
    <h2>Horizontal Federated Learning</h2>

    <p>Federated learning is a promising machine learning technique that enables multiple clients to collaboratively build a model without revealing the raw data to each other. The standard implement of FL adopts the conventional parameter server architecture where end devices are connected to the central coordinator in a start topology. The coordinator can be a central server or a base station at the network edge. Compared to traditional distributed learning methods, the key features of FL are :</p>
    <ul>
      <li><strong>No data sharing</strong> and the server stays agnostic of any training data</li>
      <li><strong>Exchange of encrypted models</strong> instead of exposing gradients</li>
      <li><strong>Sparing device-server communications</strong> instead of batch-wise gradient uploads</li>
    </ul>

    <p>Among various types of federated learning methods, horizontal federated learning (HFL) is the best-studied category and handles homogeneous feature spaces. The figure shows a typical cross-device FL scenario where a multitude of user devices are coordinated by a central server(at the edge or on the cloud). This article aims to show how horizontal federated learning can be applied to a dataset. I conducted experiments on various cases using CIFAR10 datasets and demonstrated that HFL can achieve excellent performance while ensuring the confidentiality of our data, making it a valuable tool for boosting model performance.</p>

    <img src="../public/img/Federated/HFL_global.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <h1>Architecture of HFL</h1>

    <p>Horizontal federated learning was designed for training on distributed datasets that typically share the same feature space whilst having little or non overlap in terms of their data instances. It refers to building a model in the scenario where datasets have significant overlaps on the feature spaces($X_1$, $X_2$, ....) but not on the ID spaces. The figure below show you a guidebook of HFL.</p>

    <img src="../public/img/Federated/HFL.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <p>The standard process of FL is organised in rounds. After initialisation, each round is comprised of the following steps (the figure below show an illustration of this):</p>

    <ul>
      <li>The server selects a fraction of clients randomly to participate in this round of training.</li>
      <li>The server distributes the latest global model to the selected clients</li>
      <li>The selected clients download the global model to overwrite their local models and perform training on their local data to update the modes</li>
      <li>The selected clients upload their updated local models to the server</li>
      <li>The server agregates the local models from the clients into a new global model</li>
    </ul>

    <p>The process repeats for a preset number of rounds or until the global model attains the desired level of quality (judged from the loss or accuracy in evaluations).</p>

    <img src="../public/img/Federated/HFL_architecture.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <h1>Experimental methodology</h1>

    <p>In the remainder of this article, we will delve deeper into the different aspects of horizontal federated learning by using Tensorflow Federated Learning algorithm(TFF). We have already discussed the basic structure of this process, where each client performs local training on classic machine learning models such as Decision Trees or Keras. Then, the parameters of these models are sent to the central server to be aggregated, creating a better performing overall model. This crucial step requires the choice of an optimal aggregation function in order to obtain the best possible precision. We will therefore discuss in detail the different aggregation functions used in the federated model, as well as the tests carried out to evaluate their performance. We will also present the datasets used during these tests and interpret the results obtained.</p>

    <p>Additionally, training a federated model involves several essential compartments, such as client selection, client training, model aggregation, and updates. We will pay particular attention to these key components and discuss their role and importance in the federated learning process.</p>

    <p>Finally, we will examine the different tests carried out to improve the accuracy of the federated model. We will analyze the results obtained by varying the hyperparameters and exploring different configurations. This step will allow us to understand how hyperparameter choices affect model performance and identify best practices to improve accuracies.</p>

    <h2>Aggregation functions of TFF model</h2>

    <h3>Datasets</h3>

    <p>CIFAR-10 data is one of the most commonly used datasets in the field of computer vision and machine learning. They are widely used for evaluation and comparison of machine learning models in image classification tasks.</p>

    <p>The CIFAR-10 data consists of a set of 60,000 color images of size 32x32 pixels. The images are divided into 10 different classes, with 6,000 images per class. Classes include: <strong>plane, automobile, bird, cat, deer, dog, frog, horse, ship and truck</strong>.</p>

    <p>The CIFAR-10 dataset consists of 60,000 32 x 32 color images divided into 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The dataset is divided into five training batches and one testing batch, each containing 10,000 images. The test batch contains exactly 1000 randomly selected images from each class. Training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. In total, the training batches contain exactly 5,000 images of each class.</p>

    <p>The images are in RGB format (red, green, blue) and are represented as 3D matrices with a depth of 3 color channels. Each image is represented by a matrix of dimensions 32x32x3, where the three dimensions correspond to width, height and color channels respectively. Pixels in each color channel are represented by 8-bit unsigned integers, ranging from 0 to 255.</p>

    <p>CIFAR-10 data is often used to train and evaluate image classification models, such as convolutional neural networks (CNN). These data are popular for classification tasks due to their diversity, complexity, and relatively small size. Model performance is typically evaluated in terms of classification accuracy on the test batch.</p>

    <img src="../public/img/Federated/cifar10.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <h3>Aggregation functions</h3>

    <img src="../public/img/Federated/functions.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <h3>Evaluation of aggregate functions</h3>

    <p>Federated models use different aggregation functions to combine updates from clients' local models. In our case, we chose the following functions which are the most suitable for our needs: <code>build_weighted_fed_avg()</code>, <code>build_unweighted_fed_avg()</code>, <code>build_fed_eval()</code>, and <code>build_fed_sgd()</code>. Each of these aggregation functions has specific characteristics and can influence the performance of the overall model. It is therefore important to understand the specificities of the different functions in order to choose the most suitable function in your case.</p>

    <p>To evaluate the performance of the different aggregation functions, we carried out several tests. We varied the number of clients in each test in order to measure the impact of this parameter on the accuracies obtained. For each configuration, we noted the accuracy (correct classification rate) as well as the loss of the federated model. These metrics allow us to evaluate the quality of the model's predictions and understand how performance varies depending on the number of customers.</p>

    <p>By performing these tests, we aim to identify the aggregation function best suited to our specific use case. We seek to maximize the accuracies obtained while minimizing the loss of the federated model. By analyzing the results obtained, we will be able to determine which aggregation function offers the best performance in our particular context.</p>

    <ul>
      <li>Federated learning on 3 clients <img src="../public/img/Federated/FL_3.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;"></li>
      <li>Federated learning on 5 clients <img src="../public/img/Federated/FL_5.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;"></li>
      <li>Federated learning on 20 clients <img src="../public/img/Federated/FL_20.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;"></li>
      <li>Federated learning on 50 clients <img src="../public/img/Federated/FL_50.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;"></li>
      <li>Federated learning on 100 clients <img src="../public/img/Federated/FL_100.png" alt="architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;"></li>
    </ul>

    <h3>Results analysis of tests</h3>

    <p>The study carried out revealed that the <code>build_weighted_fed_avg()</code> method performs better when used with any number of clients, resulting in better accuracy. This function sets up a federated average calculation process that integrates customer weights during the aggregation step. Client weights are typically determined based on specific criteria, such as the number of data points or the quality of the client's data. This approach allows customer contributions to be weighted differently, placing greater importance on customers whose data is more representative or reliable.</p>

    <p>Additionally, another test was carried out using 100 clients, and it was observed that the accuracies obtained are even better. This can be explained by the fact that the higher the number of clients, the greater the chances of obtaining the best weights during aggregation. Indeed, by having a larger number of clients, it is possible to benefit from a greater diversity of data, which can improve the accuracy of the federated model. It is important to note that other criteria may also contribute to improving these accuracies, and these may vary depending on the specific context of the application.</p>

    <h2>Training components of TFF model</h2>

    <h3>Identifying the components of TFF</h3>

    <p>During training, I identified the key components of the federated learning model, which are virtual or logical entities responsible for different parts of the federated learning process:</p>

    <pre><code>
    Client 1: client_work
    Client 2: aggregator
    Client 3: finalizer
    Client 4: distributor
    </code></pre>

    <h3>Role and importance of components</h3>

    <p>The different components listed above have specific roles to carry out the entire process of federative learning.</p>

    <p><strong>Customer 0: distributor</strong><br>
    The client <strong>distributor</strong> is a logical entity responsible for distributing the weights of the initial model to the different clients.</p>

    <p><strong>Client 1: client_work</strong><br>
    The term <strong>client</strong> generally refers to decentralized entities or devices (e.g. mobile devices, edge servers, etc.) that perform learning on their own local data. Each client performs local training on its data and then sends model updates to the central server (aggregator) for aggregation.</p>

    <p><strong>Client 2: aggregator</strong><br>
    The <strong>aggregator</strong> entity is responsible for aggregating model updates from different clients. It combines updates from multiple clients to get an overall model update. The aggregator can use different aggregation strategies, such as weighted average of client updates, to update the overall model.</p>

    <p><strong>Client 3: Finalizer</strong><br>
    The <strong>finalizer</strong> client is involved in some post-processing or validation steps after model aggregation. It can perform additional calculations, quality assessments, or final adjustments on the aggregated model before deploying it or making it available for future learning iterations.</p>

    <h2>Improved performance of TFF</h2>

    <p>The training was carried out in two different cases. In the first case, the basic CNN model was used by clients to classify images. A set of hyperparameter values, such as learning rate and decay, was tested over 150 epochs to study the variation in accuracies and determine the best combinations. In a second case, the pre-trained model <strong>EfficientNet</strong> was used to compare the results using the CNN model. The graphs below illustrate these different variations.</p>

    <p>By analyzing the graphs, we can observe changes in accuracy based on hyperparameter values. This allows us to choose the best performing combinations to optimize the model. Variations in accuracies give us indications on the sensitivity of the model to different hyperparameter settings.</p>

    <p>Using these results, we can select the hyperparameter values ​​that lead to maximum accuracy for our base CNN model. This helps us fine-tune settings and improve model performance when classifying images.</p>

    <p><strong>NB</strong>: <strong>Learning rate</strong>: hyperparameter that determines the size of the steps that your learning algorithm takes when optimizing the model. In other words, it controls the speed at which your model learns.</p>

    <p><strong>Decay</strong> is a concept linked to the learning rate. This is a commonly used technique to gradually reduce the learning rate over time while training the model. Decay helps stabilize learning by adjusting the learning rate as training progresses.</p>

    <h3>Training on different hyperparameters</h3>

    <h4>Using CNN model</h4>

    <ul>
      <li>Variation of learning rate<br>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
          <img src="../public/img/Federated/CNN_lr_acc.png" alt="architecture" style="max-width:45%; height:auto;">
          <img src="../public/img/Federated/CNN_lr_loss.png" alt="architecture" style="max-width:45%; height:auto;">
        </div>
      </li>

      <li>Variation of decay<br>
        <img src="../public/img/Federated/CNN_decay_acc.png" alt="architecture" style="display: block; margin: 0 auto; max-width:45%; height:auto;">
      </li>
    </ul>

    <h4>Using EfficientNet model</h4>

    <ul>
      <li>Variation of learning rate<br>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
          <img src="../public/img/Federated/Eff_lr_acc.png" alt="architecture" style="max-width:45%; height:auto;">
          <img src="../public/img/Federated/Eff_lr_loss.png" alt="architecture" style="max-width:45%; height:auto;">
        </div>
      </li>
      <li>Variation of decay<br>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
          <img src="../public/img/Federated/Eff_decay_acc.png" alt="architecture" style="max-width:45%; height:auto;">
          <img src="../public/img/Federated/Eff_decay_loss.png" alt="architecture" style="max-width:45%; height:auto;">
        </div>
      </li>
    </ul>

    <h3>Results analysis</h3>

    <h4>Using CNN model</h4>

    <p>An increasing trend in accuracy can be observed as training progresses, but at a certain level it decreases significantly. Indeed, the accuracy is higher when the learning rate has a low value, but it decreases considerably when the learning rate is higher.</p>

    <p>From the graph, we see that the smaller the decay, the higher the precision. The tests carried out demonstrated that optimal training is between 9% and 38% accuracy, which is considerably better than previously obtained results. However, it is possible that the accuracy remains static due to the low number of clients (5 clients). By increasing the number of customers, the results could be even better. It is also important to take into account that the CNN model may have difficulty adapting to a federated model.</p>

    <h4>Using EfficientNet model</h4>

    <p>The two tests revealed that the optimal values ​​of the learning rate are between 0.01 and 0.07, while those of the decay are between 0.001 and 1E-9, of the order of 1E-1. The maximum accuracy obtained is around 78%, which is significantly higher than the accuracy obtained with the CNN model.</p>

    <p>It is interesting to note that the most optimal values ​​of learning rate and decay are generally low. This indicates that lower learning rates and decays promote better performance. The EfficientNet-based TFF model is found to be more optimal due to its ability to classify images accurately, thanks to its pre-training. Even better accuracies can be achieved by increasing the number of clients or further adjusting the parameters.</p>

    <p>In summary, the test results suggest that lower values ​​of learning rate and decay lead to better accuracies. The EfficientNet-based TFF model, as a pre-trained model, provides better initial performance. To further improve results, it is recommended to explore configurations with larger numbers of clients or continue adjusting settings.</p>

    <h1>Conclusion</h1>

    <p>This article has explored in detail the different aspects of horizontal federated learning, including its definition, architecture, feasibility and case studies. We have seen that the federated learning model has many advantages, in particular the preservation of data confidentiality, while allowing good results in terms of precision.</p>

    <p>The results of the tests carried out demonstrated that the use of the horizontal federated learning model, using techniques such as the TFF model based on EfficientNet, can lead to higher levels of accuracy than those obtained with traditional models such as the CNN. The optimal values ​​of learning rate and decay were identified, and it was found that lower values ​​of these hyperparameters promote better performance.</p>

    <p>It is now worth emphasizing that the next research works focus on vertical federated learning, which is a different case from horizontal federated learning when it comes to data usage between clients. Vertical federated learning involves collaboration between clients with different but complementary data. This approach opens new perspectives for federated learning and requires in-depth study to optimize performance and data privacy in this context.</p>

    `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'serverless' }, { id: 'ai' }],
  },
  {
    id: 'microservices',
    slug: 'fl',
    image: '../img/Federated/work_FL.png',
    alt: 'Microservices Blog',
    tag: 'AI',
    date: 'Feb 15, 2024',
    title: 'Federated Learning',
    description:
      "This article introduces federated learning, a modern approach to machine learning that preserves data privacy during model training. Unlike centralized learning, where data is collected in a single location, federated learning enables models to be trained directly on users' devices, without sharing the raw data.",
    href: '/blogs/fl',
    color: 'neon-purple',
    readTime: '8 min read',
    content: `
    <h2>Federated Learning</h2>

    <p>Federated learning is a machine learning technique that allows multiple parties to collaboratively
    train a machine learning model without sharing their private data. In traditional machine learning, all data is
    collected and centralized in a single location, such as a server or data center, where the model is trained.
    However, in federated learning, the data remains decentralized and is stored locally on devices such as
    smartphones or IoT devices.</p>

    <p>In federated learning, the model is initially trained on a small sample of data from each device, and the
    updated model is then sent back to the devices to improve its accuracy. Each device locally trains the model
    using its private data and sends the updated model weights to the central server. The server then aggregates
    the updates and uses them to improve the model. This process is repeated iteratively until the desired level of
    accuracy is achieved.</p>

    <p>Federated learning has the potential to revolutionize the way machine learning models are trained and deployed in various industries. One of its key advantages is that it allows organizations to collaborate and train machine learning models on a large amount of data without the need to centralize or share their data. This preserves data privacy and security, making it particularly useful in scenarios where the data is sensitive, such as healthcare, finance, and personal devices.</p>

    <p>The applications of federated learning are wide-ranging and diverse. It can be used in personalized recommendation systems, natural language processing, image and video recognition, and predictive maintenance. However, there are also challenges associated with federated learning. Communication and computational costs can be significant, and there is a risk of biased or inaccurate models.</p>

    <p>Despite these challenges, ongoing research and advancements in federated learning are addressing these issues. With further progress, federated learning holds great promise in enabling organizations to leverage large amounts of data for machine learning while preserving privacy and security. This has the potential to transform various industries and unlock new possibilities for machine learning applications.</p>

    <h1>What is federated learning?</h1>

    <p>Federated learning (often referred to as collaborative learning) is a decentralized approach to training machine learning models. It doesn’t require an exchange of data from client devices to global servers. Instead, the raw data on edge devices is used to train the model locally, increasing data privacy. The final model is formed in a shared manner by aggregating the local updates.</p>

    <p>Here’s why federated learning is important:</p>

    <ul>
      <li><strong>Privacy</strong>: In contrast to traditional methods where data is sent to a central server for training, federated learning allows for training to occur locally on the edge device, preventing potential data breaches.</li>
      <li><strong>Data security</strong>: Only the encrypted model updates are shared with the central server, assuring data security. Additionally, secure aggregation techniques such as Secure Aggregation Principle allow the decryption of only aggregated results.</li>
      <li><strong>Access to heterogeneous data</strong>: Federated learning guarantees access to data spread across multiple devices, locations, and organizations. It makes it possible to train models on sensitive data, such as financial or healthcare data while maintaining security and privacy. And thanks to greater data diversity, models can be made more generalizable.</li>
    </ul>

    <h1>Architecture of federated learning</h1>

    <p><img src="../public/img/Federated/architecture.png" alt="architecture" style="display: block; margin: 0 auto; max-width:60%; height:auto;"></p>

    <p>The architecture of federated learning typically consists of three main components: the client
    devices, the central server, and the machine learning model.</p>

    <p><strong>Client devices</strong>: The client devices are the endpoints that hold the local data and are used to train the machine learning model. These devices can be mobile phones, laptops, IoT devices, or any other device capable of running a machine learning algorithm. In federated learning, the data remains on the client devices, and the algorithm runs on each device locally.</p>

    <p><strong>Central server</strong>: The central server acts as a coordinator and aggregator for the training process. It is responsible for managing the training process, aggregating the model updates from the client devices, and sending the updated model back to the devices. The server can also perform additional tasks, such as initializing the model and distributing it to the client devices.</p>

    <p><strong>Machine learning model</strong>: The machine learning model is the algorithm used to learn from the data on the client devices. The model can be any type of supervised or unsupervised learning algorithm, such as neural networks, decision trees, or logistic regression.</p>

    <h1>Types of federated learning</h1>

    <p>According to the distribution features of the data, federated learning may be categorized. Assume that the data matrix <em>D<sub>i</sub></em> represents the information owned by each individual data owner, i.e., each sample and each characteristic are represented by a row and a column, respectively, in the matrix. At the same time, label data may be included in certain datasets as well. For example, we call the sample ID space <em>I</em>, the feature space <em>X</em> and the label space <em>Y</em>. When it comes to finance, labels may represent the credit of customers; when it comes to marketing, labels can represent the desire of customers to buy; and when it comes to education, labels can represent students' degrees. The training dataset includes the features <em>X</em>, <em>Y</em>, and IDs <em>I</em>. Federated learning may be classified as horizontally, vertically, or as federated transfer learning (FTL) depending on how the data is dispersed among the many parties in the feature and sample ID space. We cannot guarantee that the sample ID and feature spaces of the data parties are similar.</p>

    <h2>Federated Transfer Learning (FTL)</h2>

    <p><img src="../public/img/Federated/FTL.png" alt="FTL" style="display: block; margin: 0 auto; max-width:60%; height:auto;"></p>

    <p>Federated transfer learning is suitable while two datasets differ not only just in sample size but also in feature space. Consider a bank in China and an e-commerce firm in the United States as two separate entities. The small overlap between the user populations of the two institutions is due to geographical constraints. However, only a tiny fraction of the feature space from both companies overlaps as a result of the distinct enterprises. For example, transfer-learning may be used to generate solutions of problems for the full dataset and features under a federation. Specifically, a typical portrayal across the 2 feature spaces is learnt by applying restricted general sample sets as well as then used to produce prediction results for samples with just one-sided features. There are challenges that FTL addresses that cannot be addressed by current federated learning methods, which is why it is an essential addition to the field.</p>

    <p><em>X<sub>i</sub> ≠ X<sub>j′</sub>  Y<sub>i</sub> ≠ Y<sub>j′</sub>  I<sub>i</sub> ≠ I<sub>j</sub> ∀ D<sub>i′</sub> D<sub>j′</sub>, i ≠ j</em></p>

    <h3>Vertical Federated Learning</h3>

    <p><img src="../public/img/Federated/VFL.png" alt="VFL" style="display: block; margin: 0 auto; max-width:60%; height:auto;"></p>

    <p>Machine-learning techniques for vertically partitioned data have been suggested that preserve privacy, including gradient descent, classification, secure linear regression, association rule mining, and cooperative statistical analysis. Some studies have presented a VFL method for training a logistic regression model that preserves individual privacy. The authors investigated entity resolution and learning performance, utilizing Taylor approximation to approximate gradient and loss functions for privacy-preserving computations.</p>

    <p>In the context of VFL, or feature-based FL, two datasets may share the same sample ID space but differ in feature space. For instance, an e-commerce firm and a bank, both operating in the same city, have distinct ways of conducting business. However, their user spaces intersect significantly, as they likely include most of the region's inhabitants. While banks and e-commerce platforms track customers' income, spending habits, and credit ratings, their feature sets differ greatly.</p>

    <p>Consider a scenario where both parties aim to develop a product purchase prediction model based on product and user data. These distinct characteristics are aggregated, and the training loss and gradients are computed to create a model that incorporates data from both parties jointly.</p>

    <p>In a federated learning system, every participating party has the same identity and position, and the federated method facilitates the development of a "common wealth" plan for all involved.</p>

    <h3>Horizontal Federated Learning</h3>

    <p><img src="../public/img/Federated/HFL.png" alt="HFL" style="display: block; margin: 0 auto; max-width:60%; height:auto;"></p>

    <p>HFL can be applied in scenarios in which datasets at different sites share overlapping feature space but differ in sample space as illustrated in the figure below. It resembles the situation that data is horizontally partitioned inside a tabular view. For example, two regional banks may have very different user groups from their respective regions, and the intersection set of their users is very small. However, their business models are very similar. Hence, the feature spaces
    of their datasets are the same. Formally, we summarize the conditions for HFL as:</p>

    <p><em>X<sub>i</sub> = X<sub>j</sub>,  Y<sub>i</sub> = Y<sub>j</sub>,  I<sub>i</sub> ≠ I<sub>j</sub> ∀ D<sub>i</sub> D<sub>j</sub>, i ≠ j</em></p>

    <p>where the data feature space and label space pair of the two parties, i.e., (<em>X<sub>i</sub>, Y<sub>i</sub></em>) and (<em>X<sub>j</sub>, Y<sub>j</sub></em>) are assumed to be the same, whereas the user identifiers <em>I<sub>i</sub></em> and <em>I<sub>j</sub></em> are assumed to be different. <em>D<sub>i</sub></em> and <em>D<sub>j</sub></em> denote the datasets of the i‑th party and the j‑th party respectively.</p>

    <h1>Process of training</h1>

    <h2>Steps training</h2>

    <p>The federated learning process typically follows the following steps:</p>

    <ul>
      <li><strong>Initialization</strong>: The machine learning model is initialized on the central server and distributed to the client devices.</li>
      <li><strong>Local training</strong>: The client devices perform local training on their own data, using the machine learning algorithm.</li>
      <li><strong>Model update</strong>: After local training, the client devices send their updated model parameters to the central server.</li>
      <li><strong>Aggregation</strong>: The central server aggregates the model updates from all the client devices, using a specific aggregation strategy, such as averaging or weighted averaging.</li>
      <li><strong>Model distribution</strong>: The updated model is distributed back to the client devices, and the process starts over again.</li>
    </ul>

    <p>Federated learning can also involve multiple rounds of training, where the local training and model update steps are repeated multiple times before the final model is distributed. This process allows the model to learn from a larger dataset and converge to a more accurate solution.</p>

    <h2>How to process training?</h2>

    <p>The process of training a machine learning model involves several steps, which can vary depending on the specific algorithm and data being used. However, a general overview of the process is as follows:</p>

    <ul>
      <li><strong>Data preprocessing</strong>: The first step in training a machine learning model is to preprocess the data. This can involve tasks such as cleaning the data, transforming it into a usable format, and splitting it into training and testing sets.</li>
      <li><strong>Model selection</strong>: The next step is to select a machine learning algorithm that is suitable for the problem being addressed. This can involve evaluating the strengths and weaknesses of different algorithms, as well as considering factors such as model complexity, interpretability, and accuracy.</li>
      <li><strong>Model initialization</strong>: Once an algorithm has been selected, the model needs to be initialized with appropriate parameter values. This can involve randomly initializing the model parameters, or using a pre-trained model as a starting point.</li>
      <li><strong>Training</strong>: The training process involves updating the model parameters to minimize the difference between the predicted outputs and the true outputs for the training data. This is typically done using an optimization algorithm such as stochastic gradient descent, which adjusts the model parameters based on the gradient of the loss function.</li>
      <li><strong>Validation</strong>: During training, it is important to monitor the performance of the model on a validation set, which is a subset of the data that is not used for training. This can help to identify overfitting or underfitting, and allow for adjustments to the model.</li>
      <li><strong>Hyperparameter tuning</strong>: Machine learning models often have hyperparameters, which are settings that are not learned during training but are set before training begins. These can include learning rate, regularization strength, and the number of hidden layers in a neural network. Tuning these hyperparameters can improve the performance of the model on the validation set.</li>
      <li><strong>Testing</strong>: Once training is complete, the final model is evaluated on a separate testing set to estimate its generalization performance on new, unseen data.</li>
      <li><strong>Deployment</strong>: The final step is to deploy the trained model in a production environment, where it can be used to make predictions on new data. This can involve integrating the model into a software system or deploying it as a web service.</li>
    </ul>

    <h1>Tools for federated learning</h1>

    <p>There are several tools and frameworks available for implementing federated learning, some of which are:</p>

    <ul>
      <li><strong>TensorFlow Federated</strong>: TensorFlow Federated (TFF) is an open-source framework developed by Google that enables developers to implement federated learning using TensorFlow, a popular machine learning library. TFF provides a set of APIs for building and training federated learning models.</li>
      <li><strong>PySyft</strong>: PySyft is an open-source framework developed by OpenMined that enables developers to implement privacy-preserving machine learning, including federated learning. PySyft provides a set of APIs for building and training federated learning models in Python.</li>
      <li><strong>Flower</strong>: Flower is an open-source federated learning framework developed by Adap, which enables developers to build and train federated learning models using PyTorch. Flower provides a set of APIs for building and training federated learning models, as well as tools for managing federated learning workflows.</li>
      <li><strong>FedML</strong>: FedML is an open-source framework developed by Tencent that provides a set of APIs for building and training federated learning models. FedML supports multiple machine learning frameworks, including TensorFlow, PyTorch, and Keras.</li>
      <li><strong>IBM Federated Learning</strong>: IBM Federated Learning is a commercial product developed by IBM that provides a platform for building and training federated learning models. The platform supports multiple machine learning frameworks, including TensorFlow, PyTorch, and Keras.</li>
      <li><strong>NVIDIA Clara Federated Learning</strong>: NVIDIA Clara Federated Learning is a commercial product developed by NVIDIA that provides a platform for building and training federated learning models. The platform supports multiple machine learning frameworks, including TensorFlow, PyTorch, and Keras.</li>
    </ul>

    <p>The choice of federated learning tools and frameworks will depend on factors such as the specific use case, the machine learning frameworks used, and the technical expertise of the development team.</p>

    <h1>Conclusion</h1>

    <p>Through this article, we were able to see that federated learning opened the way to promising new possibilities in terms of privacy-friendly machine learning. By preserving sensitive data where it is collected, this decentralized approach enables collaboration without compromising security.
    We explained how federated learning generally works, with training local models and aggregating updates to form a powerful global model. Libraries like TensorFlow Federated are starting to democratize these techniques among developers.
    Although still at the research stage, the first use cases in personal assistants, health or finance suggest the potential of federated learning. Its challenges, such as data heterogeneity or bandwidth, remain to be resolved for large-scale deployment.
    With the exponential growth of personal data collected, this technique heralds profound changes in the world of AI. By combining the advantages of collaborative learning with absolute respect for privacy, federated learning could become essential. Its future progress will ensure the responsible development of artificial intelligence.</p>

    `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'serverless' }, { id: 'ai' }],
  },
  {
    id: 'microservices',
    slug: '3d',
    image: '../img/3D/index.jpg',
    alt: 'Microservices Blog',
    tag: 'AI',
    date: 'Dec 04, 2023',
    title: '3D Generative Adversial Model',
    description:
      "This article introduces generative adversarial networks (GANs), a state-of-the-art deep learning technique. GANs use two neural networks in an adversarial game to generate synthetic data, with applications in image, audio, and video synthesis. The aim is to provide an overview of GANs' generative process and their immense potential.",
    href: '/blogs/3d',
    color: 'neon-purple',
    readTime: '8 min read',
    content: `
    <h2>3D Generative Adversial Model</h2>

    <div class="section">
    <h1>Introduction</h1>
    <p>Generative adversarial networks, or GANs, represent a cutting-edge approach to generative modeling in deep learning, often leveraging architectures such as convolutional neural networks. The goal of generative modeling is to autonomously identify patterns in the input data, allowing the model to produce new examples that realistically resemble the original dataset.</p>
    <p>GANs address this challenge through a unique setup, treating it as a supervised learning problem involving two key elements: the generator, which learns to produce new examples, and the discriminator, responsible for distinguishing between real and generated. Through adversarial training, these models engage in competitive interaction until the generator becomes adept at creating realistic samples, fooling the discriminator about half the time.</p>
    <p>This dynamic field of GANs has evolved rapidly, exhibiting remarkable capabilities in generating realistic content in various domains. Notable applications include image-to-image translation tasks and the creation of photorealistic images indistinguishable from real photos, demonstrating the transformative potential of GANs in the field of generative modeling.</p>
    </div>

    <div class="section">
      <h2>What is GAN model?</h2>
      <img src="../public/img/3D/globalGANModel.png" alt="Global GAN Model">
      <p><strong>GAN</strong> is a machine learning model in which two <strong>neural networks</strong> compete with each other by using <em>deep learning</em> methods to become more accurate in their predictions. GANs typically run unsupervised and use a cooperative <em>zero-sum game framework</em> to learn, where one person's gain equals another person's loss.</p>
      <p>GANs consist of two models: the <strong>generative model</strong> and the <strong>discriminator model</strong>. The generator creates fake data resembling training data, while the discriminator classifies data as real or fake. They compete until the generator produces realistic data.</p>
      <img src="../public/img/3D/GANProcess.png" alt="GAN Process" style="display: block; margin: 0 auto; max-width:60%; height:auto;">
    </div>

    <div class="section">
      <h2>How does GAN Model work?</h2>
      <p>GANs consist of two neural networks:</p>
      <ul>
        <li><strong>Generator:</strong> Learns to generate fake data that looks real</li>
        <li><strong>Discriminator:</strong> Learns to distinguish real from fake</li>
      </ul>
      <blockquote><em>The goal of the generator is to fool the discriminator, while the discriminator aims to correctly identify fake data.</em></blockquote>
      <p>The keep compete between each other until at the end fakes (generator by generator) look real (discriminator can’t differentiate).</p>
      <img src="../public/img/3D/GANDiagram.png" alt="GAN Diagram">
      <p>We notice that what we input to generator is Noise, why? Noise in this scenario, we can think about it as random small number vector. When we vary the noise on each run(training), it helps ensure that generator will generate different image on the same class on the same class based on the data that feed into discriminator and got feed back to the generator.</p>
      <img src="../public/img/3D/noiseGenerator.png" alt="Noise Generator" style="display: block; margin: 0 auto; max-width:70%; height:auto;">
      <p>Then, generate will likely generate the object that are common to find features in the dataset. For example, 2 ears with round eye of cat rather with common color rather than sphinx cat image that might pretty be rare in the dataset.</p>
      <img src="../public/img/3D/ganNetwork.png" alt="GAN Network" style="display: block; margin: 0 auto; max-width:70%; height:auto;">
      <p>
      The generator model generated images from <strong>random noise(z)</strong> and then learns how to generate realistic images.
      Random noise which is input is sampled using uniform or normal distribution and then it is fed into the generator which generated an image.
      The generator output which are fake images and the real images from the training set is fed into the discriminator that learns how to differentiate fake images from real images.
      The output <strong>D(x)</strong> is the probability that the input is real.
      If the input is real, <strong>D(x)</strong> would be 1 and if it is generated, <strong>D(x)</strong> should be 0.
    </p>

    </div>

    <div class="section">
      <h1>Metrics of GAN models</h1>

      <p><strong>1. Kullback–Leibler and Jensen–Shannon Divergence</strong></p>

      <p>Let us talk about two metrics for quantifying the similarity between two probability distributions.</p>

      <p>(1) <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">KL (Kullback–Leibler) divergence</a> measures how one probability distribution <em><strong>p</strong></em> diverges from a second expected probability distribution <strong><em>q</em></strong>.</p>

      <img src="../public/img/3D/form1.jpg" alt="GAN Network" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

      <p>D(KL) achieves the minimum zero when <strong><em>p(x) == q(x)</em></strong> everywhere. It is noticeable according to the formula that KL divergence is asymmetric. In cases where <strong><em>p(x)</em></strong> is close to zero, but <strong><em>q(x)</em></strong> is significantly non-zero, the <strong><em>q’s</em></strong> effect is disregarded. It could cause buggy results when we just want to measure the similarity between two equally important distributions.</p>

      <p>(2) <a href="https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence">Jensen–Shannon Divergence</a> is another measure of similarity between two probability distributions, bounded by [0,1]. JS divergence is symmetric and more smooth. <br>
      <img src="../public/img/3D/graph_KL.png" alt="graph KL" style="display: block; margin: 0 auto; max-width:50%; height:auto;" /></p>

      <img src="../public/img/3D/form2.jpg" alt="graph KL" style="max-width:50%; height:auto;" /></p>

      <p>Some believe (<a href="https://arxiv.org/pdf/1511.05101.pdf">Huszar, 2015</a>) that one reason behind GANs’ big success is switching the loss function from asymmetric KL divergence in traditional maximum-likelihood approach to symmetric JS divergence.</p>

      <p><u><b><strong>Small explanation about these metrics with examples</strong></b></u></p>

      <p>Let's take the example of an image generator that creates images of cats. You have a generator that generates cat images, and you also have a set of real cat images.</p>

      <p>The KL (Kullback-Leibler) divergence metric quantifies the difference between the distribution of images generated by the generator and the distribution of real images. Let's assume you have two distributions: one corresponds to the distribution of images generated by the generator (let's call it P), and the other corresponds to the distribution of real images (let's call it Q). The KL divergence between P and Q measures how much information, on average, is needed to represent the differences between these two distributions. A higher value would indicate that the images generated by the generator are very different from the real images.</p>

      <p>The KL divergence is not symmetric, which means that KL(P||Q) is not the same as KL(Q||P). This means that the way the generator approaches real images may be different from the way real images approach generated images. For example, it is possible for the generator to produce images that don't resemble real images at all, while real images may have similarities with the generated images.</p>

      <p>On the other hand, the Jensen-Shannon (JS) divergence is a symmetric measure that compares the similarity between the two distributions. It uses the KL divergence to calculate a symmetric similarity measure. In other words, the JS divergence between P and Q is the same as the JS divergence between Q and P. A JS divergence value close to zero would indicate that the images generated by the generator are very similar to the real images.</p>

      <p>By using the JS divergence, you can evaluate the performance of your generator by measuring the similarity between the generated images and the real images. If the JS divergence is low, it means that the generator is capable of producing images that are similar to the real images. If the JS divergence is high, it indicates that the images generated by the generator are very different from the real images.</p>

      <p>In summary, the KL divergence measures the difference between the distributions of generated and real images, while the JS divergence measures the similarity between these distributions. These measures help you evaluate the performance of your generator by comparing it to the real objects you want to generate.</p>

      <p><strong>2. EMD (earth's mover distance) or Wassertein distance for WGAN model</strong></p>

      <p>The Earth Mover's Distance (EMD) is a method to evaluate dissimilarity between two multi-dimensional distributions in some feature space where a distance measure between single features, which we call the <em>ground distance</em> is given.</p>

      <p>The Wasserstein distance metric has several advantages over KL and JS divergences.</p>

      <ul>
      <li>First, it is more stable and often facilitates the convergence of GAN model training. It also makes it possible to better take into account mass differences between distributions, which can be useful when the distributions have different images or modes.</li>
      <li>The Wasserstein distance metric has an interesting geometric interpretation. It can be thought of as the minimum amount of work required to move mass from one distribution to another, where each unit of mass is considered a "pile of dirt" and the cost of moving is determined by a cost function. This geometric interpretation gives it interesting properties in terms of stability and convergence.</li>
      </ul>

      <p><u><b>Small explanation about wasserstein metric with examples</b></u></p>

      <p>Suppose you have a random number generator and you want to compare it to a real distribution of numbers. Your generator produces random numbers, but you want to know how similar these numbers are to those in the real distribution.</p>

      <p>The KL divergence would tell you how different the two distributions are. It would measure the amount of additional information needed to represent the differences between the two distributions. For example, if your generator primarily produces numbers between 0 and 10, while the actual distribution is centered around 100, the KL divergence would be high to indicate that the two distributions are very different.</p>

      <p>JS divergence, on the other hand, would tell you how similar the two distributions are. If your generator produces numbers that closely resemble those in the real distribution, the JS divergence would be small, indicating high similarity between the two distributions.</p>

      <p>Now let's look at the Wasserstein distance metric. She would tell you how much <strong>work</strong> is required to turn one distribution into another. In our example, this would mean how much effort you would have to put into transforming the distribution of numbers produced by your generator into the actual distribution of numbers. If the two distributions are very different, that would mean it would take a lot of work to make them similar.</p>

      <p>To illustrate this, imagine that the actual distribution of numbers is a bell-shaped curve centered around 100. Your generator, on the other hand, mainly produces numbers between 0 and 10. The Wasserstein distance metric could tell you how many earth would need to be moved to transform the flat line between 0 and 10 into a curve of 100. The higher the Wasserstein distance metric, the more work would be required to perform this transformation. Look at the following figure to visualize what i am saying. <br>
      <img src="../public/img/3D/Wasserstein1.png" alt="wasserstein" /></p>

      <h1>Types of GAN models</h1>

      <h2>Deep Convolutional Generative Adversial Network</h2>
      <img src="../public/img/3D/DCGAN.png" alt="noise generator" />

      <p>DCGAN stands for Deep Convolutional Generative Adversarial Network. It is a type of GAN that uses convolutional layers in both the generative and discriminative models.</p>

      <p>In a DCGAN, the generative model, G, is a deep convolutional neural network that takes as input a random noise vector, z, and outputs a synthetic image. The goal of G is to produce synthetic images that are similar to the real images in the training data. The discriminative model, D, is also a deep convolutional neural network that takes as input an image, either real or synthetic, and outputs a probability that the image is real. The goal of D is to correctly classify real images as real and synthetic images as fake.</p>

      <p>The overall loss function for a DCGAN is defined as the sum of the loss functions for G and D. The loss function for G is defined as:</p>

      <img src="../public/img/3D/form1.jpg" alt="noise generator" style="display: block; margin: 0 auto; max-width:50%; height:auto;" />

      <p>This loss function encourages G to produce synthetic images that are classified as real by D. In other words, it encourages G to generate images that are similar to the real images in the training data.</p>

      <p>The loss function for D is defined as:</p>

      <img src="../public/img/3D/form2.jpg" alt="noise generator" style="display: block; margin: 0 auto; max-width:50%; height:auto;" />

      <p>This loss function encourages D to correctly classify real images as real and synthetic images as fake. In other words, it encourages D to accurately differentiate between real and fake images.</p>

      <p>The overall loss function for the DCGAN is then defined as:</p>

      <img src="../public/img/3D/form3.jpg" alt="noise generator" style="display: block; margin: 0 auto; max-width:50%; height:auto;" />

      <p>This loss function is minimized during training by updating the weights of G and D using gradient descent. By minimizing this loss function, the DCGAN learns to generate high-quality synthetic images that are similar to the real images in the training data.</p>

      <h2>Wasserstein GAN</h2>
      <img src="../public/img/3D/Wasserstein.png" alt="noise generator" />

      <p><strong>Wasserstein GANs (WGANs)</strong> are a type of Generative Adversarial Network (GAN) that use the Wasserstein distance (also known as the Earth Mover’s distance) as a measurement between the generated and real data distributions, providing several advantages over traditional GANs, which include improved stability and more reliable gradient information.</p>

      <p>The architecture of a WGAN is not different than the traditional GAN, involving a generator network that produces fake images and a discriminator network that distinguishes between real and fake images. However, instead of using a binary output for the discriminator, a WGAN uses a continuous output that estimates the Wasserstein distance between the real and fake data distributions. During training, the generator is optimized to minimize the Wasserstein distance between the generated and real data distributions, while the discriminator is optimized to maximize this distance, leading to a more stable training process. It is worth mentioning that Wasserstein distance provides a smoother measure of distance than the binary cross-entropy used in traditional GANs.</p>

      <p>One of the main advantages of WGANs is that they provide more reliable gradient information during training, helping to avoid problems such as vanishing gradients and mode collapse. In addition, the use of the Wasserstein distance provides a clearer measure of the quality of the generated images, as it directly measures the distance between the generated and real data distributions.</p>

      <p>WGANs have been used in various applications, including image synthesis, image-to-image translation, and style transfer along with additional techniques such as gradient penalty, which improves stability and performance.</p>

      <p>However, some challenges are associated with using WGANs, particularly related to the computation of the Wasserstein distance and the need for careful tuning of hyperparameters. There are also some limitations to the Wasserstein distance as a measure of distance between distributions, which can impact the model’s performance in certain situations.</p>

      <h2>CycleGANs</h2>

      <img src="../public/img/3D/CycleGan.png" alt="noise generator" />

      <p>CycleGANs are a Generative Adversarial Network (GAN) used for image-to-image translation tasks, such as converting an image from one domain to another. Unlike traditional GANs, CycleGANs do not require paired training data, making them more flexible and easier to apply in real-world settings.</p>

      <p>The architecture of a CycleGAN consists of two generators and two discriminators. One generator takes as input an image from one domain and produces an image in another domain whereas the other generator takes as input the generated image and produces an image in the original domain. The two discriminators are used to distinguish between real and fake images in each domain. During training, the generators are optimized to minimize the difference between the original image and the produced image by the other generator, while the discriminators are optimized to distinguish between real and fake images correctly. This process is repeated in both directions, creating a cycle between the two domains.</p>

      <p>CycleGANs do not require paired training data which makes them more flexible and easier to apply in real-world settings. For example, they can be used to translate images from one style to another or generate synthetic images similar to real images in a particular domain.</p>

      <p>CycleGANs have been used in various applications, including image style transfer, object recognition, and video processing. Additionally, they are also used to generate high-quality images from low-quality inputs, such as converting a low-resolution image to a high-resolution image.</p>

      <p>However, CycleGANs come with certain challenges like complexity of the training process and the need for careful tuning of hyperparameters. In addition, there is a risk of mode collapse, where the generator produces a limited set of images that do not fully capture the diversity of the target domain.</p>

      <h1>Conclusion</h1>

      <p>In this article, we presented GANs, a new type of deep learning model capable of generating realistic contents such as images, text or video. After having defined the general operation of GANs composed of a generator and a discriminator confronting each other, we detailed some architectures such as basic GANs, conditional GANs or introspective GANs. We also looked at the main challenges related to unstable training of GANs, as well as their applications in areas like image synthesis or machine translation. Although perfectible, GANs open the way to creative artificial intelligence capable of generating new content autonomously. Future progress should enable ever more realistic generations and new innovative applications.</p>

    </div>
    `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'serverless' }, { id: 'ai' }],
  },
  {
    id: 'microservices',
    slug: 'TransformerNLP',
    image: '../img/transformer/index.jpg',
    alt: 'Microservices Blog',
    tag: 'AI',
    date: 'Oct 06, 2023',
    title: 'Understanding the transformer architecture in NLP',
    description:
      "This article gives an in-depth overview of the Transformer architecture, which has revolutionized natural language processing. It focuses on attention blocks, the key component of the model that establishes parallel and contextual connections between words in a sentence.",
    href: '/blogs/TransformerNLP',
    color: 'neon-purple',
    readTime: '8 min read',
    content: `
    <h1>Neural style transfer</h1>

    <p>Natural Language Processing (NLP) is a rapidly growing field of artificial intelligence. Being able to understand, generate and translate text effectively are challenges that open the way to many technological advances.
    For a long time, Long Short-Term Memory5 (LSTM) based on Recurrent Neural Networks (RNN) were the dominant approach for sequential modeling of language data. LSTMs are better than traditional RNNs at retaining information over long sequences thanks to their gate mechanism.
    However, their ability to effectively capture very long-range contextual dependencies remains limited. It is in this context that the Transformer model appeared in 2017, proposed by a team of Google researchers.
    Rather than using a recurrent structure, Transformers incorporate an attention mechanism allowing them to learn the contextual dependencies between elements in a sequence. This revolutionary architecture very quickly surpassed RNNs on many NLP tasks such as machine translation.
    Since then, Transformers have become omnipresent in the field. Giant models such as BERT or GPT-3 have enabled major advances in understanding and generating text. However, many questions remain open about their complex inner workings.
    In this article, we present in detail the Transformer architecture as well as its current applications.</p>

    <h1>Sequentials models of NLP</h1>

    <h2>What is RNN?</h2>

    <p><strong>Recurrent neural networks (RNN)</strong> are models specialized in the analysis of sequential data such as text or speech.
    Unlike traditional networks which only see information isolated from each other, RNNs are able to <strong>memorize</strong> what they have already seen thanks to their internal memory.
    This memory, called <strong>hidden state</strong>, keeps track of the previous context at each stage of processing a sequence. So when the RNN looks at a new element, it also remembers the previous ones thanks to its hidden state.
    This is what allows RNNs to efficiently analyze data like sentences or music, where the order of words/sounds is important. Rather than seeing everything separately, the RNN understands how each part fits together.
    Thanks to their dynamic internal memory, RNNs are today widely used in language and speech processing by machines. It is one of the key tools to teach them to communicate better with us. The figure below represent a global architecture of RNN where x, h, o are the input sequence, hidden state and output sequence respectively. U, V and W are the training weights.</p>

    <img src="../public/img/transformer/RNN.png" alt="RNN Architecture">

    <p>However, they face a limitation called the <strong>vanishing gradient problem</strong>. Indeed, when an RNN processes the elements of a sequence one after the other, the influence of the first elements analyzed tends to fade over time. It's as if the network has more and more difficulty remembering the beginning of the sequence as it goes on. Then, <strong>LSTM</strong> model come to resolve it.</p>

    <h2>What is LSTM ?</h2>

    <p>LSTM is a specific type of RNN architecture that addresses the vanishing gradient problem, which occurs when training deep neural networks. LSTMs leverage memory cells and gates to selectively store and retrieve information over long sequences, making them effective at capturing long-term dependencies. The figure blow shows a memory cell architecture of LSTM model:</p>

    <img src="../public/img/transformer/LSTM.png" alt="LSTM Architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <p>LSTMs have a <strong>memory cell</strong> allowing them to better manage long-term dependencies. This memory cell is made up of three <strong>gates</strong>:</p>

    <ul>
        <li>The <strong>input gate</strong></li>
        <li>The <strong>forget gate</strong></li>
        <li>The <strong>output gate</strong></li>
    </ul>

    <p>These gates regulate the flow of information inside the memory cell, thus making it possible to control what information is remembered and what information is forgotten. This gives LSTM the ability to remember important information over long sequences and ignore less relevant material. $h_t$ is the usual hidden state of RNNs but in LSTM networks we add a second state called $c_t$. Here, $h_t$ represents the neuron's short memory (previous word) and $c_t$ represents the long-term memory (all the previous words history).</p>

    <h3>Forget gate</h3>
    <img src="../public/img/transformer/porte-doubli-LSTM.gif" alt="Forget Gate">
    <p>This gate decides what information must be kept or discarded: the information from the previous hidden state is concatenated to the input data (for example the word "des" vectorized) then the sigmoid function is applied to it in order to normalize the values between 0 and 1. If the output of the sigmoid is close to 0, this means that we must forget the information and if it is close to 1 then we must memorize it for the rest.</p>

    <h3>Input gate</h3>
    <img src="../public/img/transformer/inputGate.gif" alt="Input Gate">
    <p>The role of the entry gate is to extract information from the current data (the word “des” for example): we will apply in parallel a sigmoid to the two concatenated data (see previous gate) and a tanh.</p>
    <ul>
        <li>Sigmoid (on the blue circle) will return a vector for which a coordinate close to 0 means that the coordinate in the equivalent position in the concatenated vector is not important. Conversely, a coordinate close to 1 will be deemed “important” (i.e. useful for the prediction that the LSTM seeks to make).</li>
        <li>Tanh (on the red circle) will simply normalize the values ​​(overwrite them) between -1 and 1 to avoid problems with overloading the computer with calculations.</li>
        <li>The product of the two will therefore allow only the important information to be kept, the others being almost replaced by 0.</li>
    </ul>

    <h3>Cell state</h3>
    <img src="../public/img/transformer/cellState.gif" alt="Cell State">
    <p>We talk about the state of the cell before approaching the last gate (output gate), because the value calculated here is used in it.
    The state of the cell is calculated quite simply from the oblivion gate and the entry gate: first we multiply the exit from oblivion coordinate by coordinate with the old state of the cell. This makes it possible to forget certain information from the previous state which is not used for the new prediction to be made. Then, we add everything (coordinate to coordinate) with the output of the input gate, which allows us to record in the state of the cell what the LSTM (among the inputs and the previous hidden state) has deemed relevant.</p>

    <h3>Output gate</h3>
    <img src="../public/img/transformer/outputGate.gif" alt="Output Gate">
    <p>Last step: the output gate must decide what the next hidden state will be, which contains information about previous inputs to the network and is used for predictions.
    To do this, the new state of the cell calculated just before is normalized between -1 and 1 using tanh. The concatenated vector of the current input with the previous hidden state passes, for its part, into a sigmoid function whose goal is to decide which information to keep (close to 0 means that we forget, and close to 1 that we will keep this coordinate of the state of the cell).
    All this may seem like magic in the sense that it seems like the network has to guess what to retain in a vector on the fly, but remember that a weight matrix is ​​applied as input. It is this matrix which will, concretely, store the fact that such information is important or not based on the thousands of examples that the network will have seen!</p>

    <h2>What is a Transformer?</h2>
    <p>The Transformer is a neural network architecture proposed in the seminal paper “Attention Is All You Need” by Vaswani et al. Unlike RNNs, Transformers do not rely on recurrence but instead operate on self-attention.
    Self-attention allows the model to weigh the importance of different input tokens when making predictions, enabling it to capture long-range dependencies without the need for sequential processing. Transformers consist of encoder and decoder layers, employing multi-head self-attention mechanisms and feed-forward neural networks.
    The figure below shows the architecture of a Transformer network:</p>

    <img src="../public/img/transformer/The-transformer-model-architecture.png" alt="Transformer Architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <h2>From LSTM to transformers</h2>

    <p>Neural networks are very efficient statistical models for analyzing complex data with variable formats.
    If models like CNNs emerged for processing visual data, for text processing, the neural network architectures that were used were RNNs, and more particularly LSTMs.</p>

    <p>LSTMs make it possible to resolve one of the major limitations of classic neural networks: they make it possible to introduce a notion of context, and to take into account the temporal component.
    This is what made them popular for language processing. Instead of analyzing words one by one, we could analyze sentences in a very specific context.
    However, LSTMs, and RNNs in general, did not solve all the problems.</p>

    <ul>
      <li>First, their memory is too short to be able to process paragraphs that are too long</li>
      <li>Then, RNNs process data sequentially, and are therefore difficult to parallelize. Except that the best language models today are all characterized by an astronomical amount of data. Training an LSTM model using the data consumed by GPT-3 would have taken decades.</li>
    </ul>

    <p>This is where Transformers come in, to revolutionize deep learning. They were initially proposed for translation tasks. And their major characteristic is that they are easily parallelizable. Which makes training on huge databases faster.
    For example, <a href="https://fr.wikipedia.org/wiki/GPT-3">GPT-3</a> was trained on a database of over 45TB of text, almost the entire internet. They have made it possible to achieve unprecedented levels of performance on tasks such as translation or image generation and are the basis of what we today call <a href="https://larevueia.fr/introduction-a-lintelligence-artificielle-generative/">generative artificial intelligence</a>.</p>

    <p>Let's study <strong>GPT transfomer architecture</strong> 😊</p>

    <h3>Transformer architecture of GPT model</h3>

    <img src="../public/img/transformer/transformer_GPT.png" alt="GPT Architecture" style="display: block; margin: 0 auto; max-width:50%; height:auto;">

    <p>In this diagram, the data flows from the bottom to the top, as is traditional in Transformer illustrations. Initially, our input tokens undergo several encoding steps:</p>
    <ul>
      <li>They are encoded using an Embedding layer. This assigns a unique vector representation to each input token.</li>
      <li>They then pass through a Positional Encoding layer. This encodes positional information by adding signals to the embedding vectors.</li>
      <li>The output of the Embedding layer and Positional Encoding layer are added together. This combines the token representation with its positional context.</li>
    </ul>

    <p>Next, the encoded inputs go through a sequence of N decoding steps. Each decoding step processes the encoded inputs using self-attention and feedforward sublayers.</p>
    <p>Finally, the decoded data is processed in two additional steps:</p>
    <ul>
      <li>It passes through a normalization layer to regulate the output scale.</li>
      <li>It is then sent through a linear layer and softmax. This produces a probability distribution over possible next tokens that can be used for prediction.</li>
    </ul>

    <p>In the sections that follow, we’ll take a closer look at each of the components in this architecture.</p>

    <h2>Embedding</h2>

    <p>The Embedding layer turns each token in the input sequence into a vector of length <strong>d_model</strong>. The input of the Transformer consists of batches of sequences of tokens, and has shape <strong>(batch_size, seq_len)</strong>. The Embedding layer takes each token, which is a single number, calculates its embedding, which is a sequence of numbers of length <strong>d_model</strong>, and returns a tensor containing each embedding in place of the corresponding original token. Therefore, the output of this layer has shape <strong>(batch_size, seq_len, d_model)</strong>.</p>

    <pre><code class="language-python">
    import torch.nn as nn

    class Embeddings(nn.Module):
      def __init__(self, d_model, vocab_size): 
        super(Embeddings, self).__init__() 
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model 
      def forward(self, x): 
        out = self.lut(x) * math.sqrt(self.d_model) 
        return out
    </code></pre>

    <p>The purpose of using an embedding instead of the original token is to ensure that we have a similar mathematical vector representation for tokens that are semantically similar. For example, let’s consider the words “she” and “her”. These words are semantically similar, in the sense that they both refer to a woman or girl, but the corresponding tokens can be completely different (for example, when using OpenAI’s tiktoken tokenizer, “she” corresponds to token 7091, and “her” corresponds to token 372). The embeddings for these two tokens will start out being very different from one another as well, because the weights of the embedding layer are initialized randomly and learned during training. But if the two words frequently appear nearby in the training data, eventually the embedding representations will converge to be similar.</p>

    <h2>Positional Encoding</h2>

    <p>The <strong>Positional Encoding layer</strong> adds information about the absolute position and relative distance of each token in the sequence.Unlike recurrent neural networks (RNNs) or convolutional neural networks (CNNs), Transformers don’t inherently possess any notion of where in the sequence each token appears. Therefore, to capture the order of tokens in the sequence, Transformers rely on a <strong>Positional Encoding</strong>.

There are many ways to encode the positions of tokens. For example, we could implement the Positional Encoding layer by using another embedding module (similar to the previous layer), if we pass the position of each token rather than the value of each token as input. Once again, we would start with the weights in this embedding chosen randomly. Then during the training phase, the weights would learn to capture the position of each token.

</p>

    <pre><code class="language-python">
    class PositionalEncoding(nn.Module):
      def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)
      def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        return self.dropout(x)
    </code></pre>

    <h2>Decoder</h2>

    <p>As we saw in the diagrammatic overview of the Transformer architecture, the next stage after the Embedding and Positional Encoding layers is the <strong>Decoder module</strong>. The Decoder consists of N copies of a Decoder Layer followed by a <strong>Layer Norm</strong>. Here’s the <strong>Decoder class</strong>, which takes a single DecoderLayer instance as input to the class initializer:</p>
    
    <pre><code class="language-python">
    class Decoder(nn.Module): 
      def __init__(self, layer, N): 
        super(Decoder, self).__init__() 
        self.layers = clones(layer, N) 
        self.norm = LayerNorm(layer.size) 
      def forward(self, x, mask): 
        for layer in self.layers: 
          x = layer(x, mask) 
        return self.norm(x)
    </code></pre>

    <p>The Layer Norm takes an input of shape <strong>(batch_size, seq_len, d_model)</strong> and normalizes it over its last dimension. As a result of this step, each embedding distribution will start out as unit normal (centered around zero and with standard deviation of one). Then during training, the distribution will change shape as the parameters <strong>a_2</strong> and <strong>b_2</strong> are optimized for our scenario.</p>
    <pre><code class="language-python">
    class LayerNorm(nn.Module): 
      def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__() 
        self.a_2 = nn.Parameter(torch.ones(features)) 
        self.b_2 = nn.Parameter(torch.zeros(features)) 
        self.eps = eps 
      def forward(self, x): 
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True) 
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    </code></pre>

    <p>The <strong>DecoderLayer</strong> class that we clone has the following architecture:</p>

    <img src="../public/img/transformer/decoderLayer.png" alt="Decoder Layer">
    <p>Here’s the corresponding code:</p>
    <pre><code class="language-python">
    class DecoderLayer(nn.Module): 
      def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__() 
        self.size = size 
        self.self_attn = self_attn 
        self.feed_forward = feed_forward 
        self.sublayer = clones(SublayerConnection(size, dropout), 2) 
      def forward(self, x, mask): 
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        return self.sublayer[1](x, self.feed_forward)
    </code></pre>

    <p>At a high level, a <strong>DecoderLayer</strong> consists of two main steps:</p>
    <ul>
      <li><strong>The attention step</strong>, which is responsible for the communication between tokens</li>
      <li><strong>The feed forward step</strong>, which is responsible for the computation of the predicted tokens.</li>
    </ul>

    <p>Surrounding each of those steps, we have <strong>residual (or skip) connections</strong>, which are represented by <strong>the plus signs</strong> in the diagram. Residual connections provide an alternative path for the data to flow in the neural network, which allows skipping some layers. The data can flow through the layers within the residual connection, or it can go directly through the residual connection and skip the layers within it. In practice, residual connections are often used with deep neural networks, because they help the training to converge better. You can learn more about residual connections in the paper <a href="https://arxiv.org/abs/1512.03385" target="_blank">Deep residual learning for image recognition</a>, from 2015. We implement these residual connections using the <code>SublayerConnection</code> module:</p>

    <pre><code class="language-python">
    class SublayerConnection(nn.Module): 
        def __init__(self, size, dropout): 
            super(SublayerConnection, self).__init__() 
            self.norm = LayerNorm(size) 
            self.dropout = nn.Dropout(dropout) 
        def forward(self, x, sublayer): 
            return x + self.dropout(sublayer(self.norm(x)))
    </code></pre>

    <p>The feed-forward step is implemented using two linear layers with a Rectified Linear Unit (ReLU) activation function in between:</p>

    <pre><code class="language-python">
    class PositionwiseFeedForward(nn.Module):

        def __init__(self, d_model, d_ff, dropout=0.1):
            super(PositionwiseFeedForward, self).__init__()
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.w_2(self.dropout(F.relu(self.w_1(x))))
    </code></pre>

    <p>The attention step is the most important part of the Transformer, so we’ll devote the next section to it😊.</p>

    <h2>Masked multi-headed self-attention</h2>

    <p>The multi-headed attention section in the previous diagram can be expanded into the following architecture:</p>

    <p><img src="../public/img/transformer/selfAttention.png" alt="multi-headed self-attention" style="display: block; margin: 0 auto; max-width:60%; height:auto;"></p>

    <p>Each multi-head attention block is made up of four consecutive levels:</p>

    <ul>
      <li>On the first level, three linear (dense) layers that each receive the queries, keys, or values </li>
      <li>On the second level, a scaled dot-product attention function. The operations performed on both the first and second levels are repeated <em>h</em> times and performed in parallel, according to the number of heads composing the multi-head attention block. </li>
      <li>On the third level, a concatenation operation that joins the outputs of the different heads</li>
      <li>On the fourth level, a final linear (dense) layer that produces the output</li>
    </ul>

    <p><a href="https://machinelearningmastery.com/the-transformer-attention-mechanism/" target="_blank">Recall</a> as well the important components that will serve as building blocks for your implementation of the multi-head attention:</p>

    <ul>
      <li><strong>The queries</strong>, <strong>keys</strong>, and <strong>values</strong>: These are the inputs to each multi-head attention block. In the encoder stage, they each carry the same input sequence after this has been embedded and augmented by positional information. Similarly, on the decoder side, the queries, keys, and values fed into the first attention block represent the same target sequence after this would have also been embedded and augmented by positional information. The second attention block of the decoder receives the encoder output in the form of keys and values, and the normalized output of the first decoder attention block as the queries. The dimensionality of the queries and keys is denoted by d(k), whereas the dimensionality of the values is denoted by d(v).</li>
      <li><strong>The projection matrices</strong>: When applied to the queries, keys, and values, these projection matrices generate different subspace representations of each. Each attention <em>head</em> then works on one of these projected versions of the queries, keys, and values. An additional projection matrix is also applied to the output of the multi-head attention block after the outputs of each individual head would have been concatenated together. The projection matrices are learned during training.</li>
    </ul>

    <h2>Generator</h2>

    <p>The last step in our Transformer is the Generator, which consists of a linear layer and a softmax executed in sequence:</p>

    <pre><code class="language-python">
    class Generator(nn.Module): 
        def __init__(self, d_model, vocab): 
            super(Generator, self).__init__() 
            self.proj = nn.Linear(d_model, vocab) 
        def forward(self, x): 
            return F.log_softmax(self.proj(x), dim=-1)
    </code></pre>

    <p>The purpose of the linear layer is to convert the third dimension of our tensor from the internal-only <code>d_model</code> embedding dimension to the <code>vocab_size</code> dimension, which is understood by the code that calls our Transformer. The result is a tensor dimension of <code>(batch_size, seq_len, vocab_size)</code>. The purpose of the softmax is to convert the values in the third tensor dimension into a probability distribution. This tensor of probability distributions is what we return to the user.</p>

    <p>You might remember that at the very beginning of this article, we explained that the input to the Transformer consists of batches of sequences of tokens, of shape <code>(batch_size, seq_len)</code>. And now we know that the output of the Transformer consists of batches of sequences of probability distributions, of shape <code>(batch_size, seq_len, vocab_size)</code>. Each batch contains a distribution that predicts the token that follows the first input token, another distribution that predicts the token that follows the first and second input tokens, and so on. The very last probability distribution of each batch enables us to predict the token that follows the whole input sequence, which is what we care about when doing inference.</p>

    <p>The Generator is the last piece of our Transformer architecture, so we’re ready to put it all together. To know how to train and implement it all together.</p>

    <h1>Difference between RNNs and Transformers</h1>

    <h2>Architecture</h2>
    <p>RNNs are sequential models that process data one element at a time, maintaining an internal hidden state that is updated at each step. They operate in a recurrent manner, where the output at each step depends on the previous hidden state and the current input.</p>

    <p>Transformers are non-sequential models that process data in parallel. They rely on self-attention mechanisms to capture dependencies between different elements in the input sequence. Transformers do not have recurrent connections or hidden states.</p>

    <h2>Handling Sequence Length</h2>
    <p>RNNs can handle variable-length sequences as they process data sequentially. However, long sequences can lead to vanishing or exploding gradients, making it challenging for RNNs to capture long-term dependencies.</p>

    <p>Transformers can handle both short and long sequences efficiently due to their parallel processing nature. Self-attention allows them to capture dependencies regardless of the sequence length.</p>

    <h2>Dependency Modeling</h2>
    <p>RNNs are well-suited for modeling sequential dependencies. They can capture contextual information from the past, making them effective for tasks like language modeling, speech recognition, and sentiment analysis.</p>

    <p>Transformers excel at modeling dependencies between elements, irrespective of their positions in the sequence. They are particularly powerful for tasks involving long-range dependencies, such as machine translation, document classification, and image captioning.</p>

    <h2>Size of the Model</h2>
    <p>The size of an RNN is primarily determined by the number of recurrent units (e.g., LSTM cells or GRU cells) and the number of parameters within each unit. RNNs have a compact structure as they mainly rely on recurrent connections and relatively small hidden state dimensions. The number of parameters in an RNN is directly proportional to the number of recurrent units and the size of the input and hidden state dimensions.</p>

    <p>Transformers tend to have larger model sizes due to their architecture. The main components contributing to the size of a Transformer model are self-attention layers, feed-forward layers, and positional encodings. Transformers have a more parallelizable design, allowing for efficient computation on GPUs or TPUs. However, this parallel processing capability comes at the cost of a larger number of parameters.</p>

    <h2>Training and Parallelisation</h2>
    <p>For RNN, we mostly train it in a sequential approach, as the hidden state relies on previous steps. This makes parallelization more challenging, resulting in slower training times.</p>

    <p>On the other hand, we train Transformers in parallel since they process data simultaneously. This parallelization capability speeds up training and enables the use of larger batch sizes, which makes training more efficient.</p>

    <h1>Conclusion</h1>
    <p>In this article, we explain the basic idea behind RNN/LSTM and Transformer. Furthermore, we compare these two types of networks from multiple aspects. We also talked about the architecture of transformers in the GPT model.<br>
    While RNNs and LSTMs were the go-to choices for sequential tasks, Transformers are proving to be a viable alternative due to their parallel processing capability, ability to capture long-range dependencies, and improved hardware utilization. However, RNNs still have value when it comes to tasks where temporal dependencies play a critical role.<br>
    In conclusion, the choice between RNN/LSTM and Transformer models ultimately depends on the specific requirements of the task at hand, striking a balance between efficiency, accuracy and interpretability.</p>

    `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'serverless' }, { id: 'ai' }],
  },
  {
    id: 'microservices',
    slug: 'NeuralStyle',
    image: '../img/NeuralStyleTrans/index.jpg',
    alt: 'Microservices Blog',
    tag: 'AI',
    date: 'Sep 03, 2023',
    title: 'Neural style transfer',
    description:
      "This article explains neural style transfer, an AI technique combining the visual content of one image with the artistic style of another. It details how convolutional neural networks capture content and style, and how iterative optimization blends the two into a new hybrid image. A clear guide to this generative deep learning approach.",
    href: '/blogs/NeuralStyle',
    color: 'neon-purple',
    readTime: '8 min read',
    content: `
    <h2>Neural style transfer</h2>

    <h1>Introduction</h1>
    <p>Generally, separate content from style in natural images is still an extremely difficult problem. However, the recent advance of DCNNs has produced powerful computer vision systems that learn to extract high-level semantic information from natural images. Therefore, we can extract the style and content from one image to another.</p>
    <p>Transferring the style from one image to another is a problem of <strong>texture transfer</strong>. In texture transfer, the goal is to synthetize a texture from a source image while constraining the texture synthesis in order to preserve the semantic content of a target image.</p>
    <p>For texture synthesis, there exist a large range of powerful non-parametric algorithms that can synthetize photorealistic natural textures by resampling the pixels of a given source texture. Therefore, a fundamental prerequisite is to find image representations that independently model variations in the semantic image content and the style in which is presented.</p>
    <p><img src="../public/img/NeuralStyleTrans/styleNeuralNetResult1.png" alt="Style Neural Network results"></p>
    <p>As we can see, the generated image is having the content of the <em><strong>Content image and style image</strong></em>. This above result cannot be obtained by overlapping the image. So the main question are: <em><strong>What is neural style transfer? how we make sure that the generated image has the content and style of the image?  how we capture the content and style of respective images?</strong></em></p>

    <h1>What is neural style transfer?</h1>
    <p><strong>Neural Style Transfer(NST)</strong> is the technique which generated an image G containing a style of image A and the content of image C.</p>
    <p>It deals with two sets of images: <strong>Content image and Style image</strong>.</p>
    <p>This technique helps to recreate the content image in the style of the reference image. It uses Neural Networks to apply the artistic style from one image to another.</p>
    <p>NST opens up endless possibilities in design, content generation, and the development of creative tools.</p>

    <h1>How does NST work?</h1>
    <p>The goal of NST is to give to the deep learning model the ability to differentiate between style and content representations. NST uses a pre-trained convolutional neural network with additional loss functions to transfer the style and then, generate a new image with the desired features.</p>
    <p>Style transfer works by activating the neurons in a specific way, so that the output image and the content image match particularly at the content level, while the style image and the desired output image should match in terms of texture and capture the same style characteristics in the activation maps.</p>
    <p>These two objectives are combined in a single loss formula, where we can control how much we care about style reconstruction and content reconstruction.</p>
    <p>The <strong>loss function</strong> in neural style transfer plays a crucial role in guiding the model to generate an image that combines both the desired style and content. We have two functions loss : <strong>content loss and style loss</strong>.</p>
    <p>The loss function is used to quantify how well the generated image matches the style and content objectives. It measures the difference between the network activations for the original content image and the generated image, as well as the difference between the network activations for the original style image and the generated image.</p>
    <p>To balance style reconstruction and content reconstruction, the loss function combines these two differences using weights. These weights control the relative importance of style reconstruction compared to content reconstruction.</p>
    <p>Optimizing the loss function involves adjusting the pixel values in the generated image to minimize the difference between the network activations for the original content image and the generated image, while also minimizing the difference between the network activations for the original style image and the generated image.</p>
    <p>By adjusting these pixel values, the model learns to reproduce the desired style characteristics in the corresponding areas of the generated image while preserving the content information from the original image.</p>
    <p>Thus, the loss function optimizes the model to generate an image that combines the desired style and content by minimizing the discrepancies between the network activations for the reference images and the generated image.</p>

    <p>Here are the required inputs to the model for image style transfer:</p>
    <ol>
      <li><strong>A Content Image</strong> –an image to which we want to transfer style to</li>
      <li><strong>A Style Image</strong> – the style we want to transfer to the content image</li>
      <li><strong>An Input Image</strong> (random) – this will contain the final blend of content and style image</li>
    </ol>

    <h2>NST basic structure</h2>
    <p>Training a style transfer model requires two networks: <strong>a pre-trained feature extractor and a transfer network</strong>.</p>
    <p>In the case of Neural Style Transfer, we use a model pre-trained on ImageNet, such as VGG using TensorFlow. Since the VGG model cannot understand images directly, it is necessary to convert them into raw pixels, afterwards feed that pixels to the model to transform them into a set of features, which is usually done by CNNs. We will see in the next section.</p>
    <p>Thus, the VGG model acts as a complex feature extractor between the input layer (where the image is fed), and the output layer (which produces the final result). To achieve style transfer, we focus on the middle layers of the model that capture essential information about the content and style of the input images.</p>
    <p>During the style transfer process, the input image is transformed into representations that emphasize image content rather than specific pixel values.</p>
    <p>Features extracted from upper layers of the model are more closely related to the image content. To obtain a representation of the style from a reference image, we analyze the correlation between the different filter responses in the model.</p>
    <p><img src="../public/img/NeuralStyleTrans/neuralArchitecture.png" alt="Style Neural Network results" style="display: block; margin: 0 auto; max-width:50%; height:auto;"></p>
    <p><em><strong>Neural Style Transfer architecture diagram according to V7Labs</strong></em></p>

    <h3>How CNN capture features in VGG model?</h3>
    <p><img src="../public/img/NeuralStyleTrans/CNN_architecture.png" alt="Style Neural Network results"></p>
    <p>The VGG model is actually a type of CNN. VGG, which stands for Visual Geometry Group, is a very popular CNN architecture widely used in computer vision tasks, especially in the field of image classification. The VGG model is composed of several stacked convolutional layers, followed by fully connected layers. These convolutional layers are responsible for extracting visual features from images.</p>
    <p>Specifically, VGG's convolutional layers are designed to analyze visual patterns at different spatial scales. Each convolutional layer uses filters that are applied to images to detect specific patterns, such as edges, textures or shapes.</p>
    <p>The figure shows an exemple of CNN layers of VGG model. The first convolutional layers of VGG (those at level 1 with 32 filters) capture low-level features, such as simple edges and textures, while the deeper convolutional layers (those at level 2 with 64 filters) capture features of higher level like complex shapes and overall structures.</p>
    <p>Thus the VGG model as a CNN is able to extract meaningful visual features from images. These features can then be used in different tasks, such as image classification or style transfer. In the context of style transfer, the model is used primarily as a feature extractor. VGG's convolutional layers are leveraged to capture content and style information from input images, allowing these two aspects to be separated and combined to generate a new image that combines the content of a reference image and style from another image.</p>

    <h3>Content loss</h3>
    <p>Content loss is a metric that helps to establish similarities between the content image and the image generated by the style transfer model. The idea behind it is that the higher layers of the model focus more on the features present in the image, i.e. the overall content of the image. The calcul of content loss is more easy because by only working with grayscale images when calculating content loss, one focuses only on the structure and arrangement of shapes/objects in the image, without considering color or other stylistic elements.</p>
    <p>The content loss is calculated using the MSE between the upper-level intermediate feature representations of the generated image (x) and the content image (p) at layer <code>l</code>.</p>
    <p><code>L<sub>content</sub>(p,x,l) = (1/2) Σ<sub>i,j</sub> (F<sup>l</sup><sub>ij</sub>(x) - P<sup>l</sup><sub>ij</sub>(p))²</code></p>
    <p>In this equation, <code>F<sup>l</sup><sub>ij</sub>(x)</code> represents the feature representation of the generated image x at layer l and <code>P<sup>l</sup><sub>ij</sub>(p)</code> represents the representation of characteristics of the content image p at layer l.</p>

    <h3>Style loss</h3>
    <p><img src="../public/img/NeuralStyleTrans/styleLoss.png" alt="Style Neural Network results"></p>
    <p><strong>Style loss</strong> is a bit more complicated that content loss because style is hard to define exactly. It is not possible to simply compare the features like patterns, contours, shapes of the two images to obtain the style loss. We need to find the <strong>correlation</strong> between features. That is why we use another tool called: <strong>Gram matrix</strong>. The Gram matrix then captures the correlations between style characteristics. It measures how visual patterns co-occur in the image (colors, textures). More precisely, each case of the Gram matrix corresponds to the scalar product between two column vectors of the feature matrix. This allows you to obtain a signature of the image style in the form of a matrix.</p>
    <p>The Gram matrix has 2 specificities:</p>
    <ul>
      <li><strong>The Gram matrix does not take into account the position of the features</strong></li>
    </ul>
    <p>The content loss calculation takes into account the position of each pixel in order to reproduce the content of the original image in the generated image. Conversely, the loss of style is more about textures, colors and other overall aspects independent of position. This is why the Gram matrix is ​​used to capture the stylistic features present in the image.</p>
    <p>The first layers of a neural network encode features such as colors and textures. One might think that focusing on these layers as with the loss of content would result in a "loss of style". However, activation maps encode both the characteristics present but also their precise location.</p>
    <p>This is where the Gram matrix comes in useful. By eliminating the spatial component, it focuses only on feature types without considering their position. Since the objective of style transfer is to reproduce global patterns and textures rather than local details, this representation without spatial location is better suited. It highlights correlations between features regardless of their position in the image.</p>
    <ul>
      <li><strong>The Gram matrix take the correlations of two features</strong></li>
    </ul>
    <p>When a neural network analyzes an image, each neuron in a layer will be specialized in detecting a particular visual pattern such as lines, circles or squares. The strength of activation of a neuron will then indicate the presence of this pattern. However, style depends not only on the presence or absence of individual patterns, but also on how they interact with each other.</p>
    <p>This is where the Gram matrix comes into play in a relevant way. Indeed, we have seen that it does not take into account the position of features, placing much more emphasis on textures, colors and other overall aspects of the style. Additionally, it makes it possible to quantify the correlation between the activations of different neurons, revealing the extent to which two patterns tend to appear together consistently across the entire image.</p>
    <p>This information on the relationships between visual patterns then makes it possible to define the style globally and independently of the precise position of each element. During style transfer, the objective is precisely to match these global patterns between the source and target image, rather than local details. By offering a representation focused on the relationships between characteristics, the Gram matrix thus facilitates comparison and guidance of the transfer process.</p>
    <p>These two caracteristics of Gram matrix enables to retrieve style of an image by calculate style loss. So the style loss is calculated by the distance between the gram matrices (or, in other terms, style representation) of the generated image and the style reference image.</p>
    <p>The contribution of each layer in the style information is calculated by the formula below:</p>
    <p><code>E<sub>l</sub> = (1 / 4N<sub>l</sub><sup>2</sup>M<sub>l</sub><sup>2</sup>) Σ<sub>i,j</sub>(G<sub>ij</sub><sup>l</sup> - A<sub>ij</sub><sup>l</sup>)²</code> where G<sub>ij</sub><sup>l</sup> is for style image and A<sub>ij</sub><sup>l</sup> for generated image</p>
    <p>Thus, the total style loss across each layer is expressed as:</p>
    <p><code>L<sub>style</sub>(a, x) = Σ<sub>l ∈ L</sub> w<sub>l</sub> E<sub>l</sub></code></p>

    <h3>Total Loss</h3>
    <p>The total loss function is the sum of the cost of the content and the style image. Mathematically,it can be expressed as :</p>
    <p><code>L<sub>total</sub>(p, α, x) = α L<sub>content</sub>(p, x) + β L<sub>style</sub>(α, x)</code></p>
    <p>You may have noticed Alpha and Beta above. They are used for weighting Content and Style cost respectively. In general, they define the weightage of each cost in the Generated output image.</p>
    <p>Once the loss is calculated, then this loss can be minimized using backpropagation which in turn will optimize our randomly generated image into a meaningful piece of art.</p>

    <h1>Conclusion</h1>
    <p>In this article, we were able to discover a fascinating application of deep learning with neural style transfer. By separating the content and style of different works, models like Neural Style are able to combine their respective styles in stunning ways.</p>
    <p>Although still imperfect, the results obtained using pre-trained networks like VGG demonstrate the potential of this approach to generate new, never-before-seen artistic creations. Beyond the fun aspect, style transfer also opens up perspectives for image editing and retouching.</p>
    <p>We have seen that current work attempts to refine the separation of content and style, or to extend these techniques to other media such as video. In the future, more advanced models could further assist human creativity.</p>
    <p>But beyond the applications, style transfer above all illustrates the astonishing capacity that artificial intelligence has to understand and imitate complex visual styles, thanks to recent advances in deep learning.</p>

    `,
    author: {
      name: 'Djoupe Audrey',
      role: 'Full-Stack Futurist',
      image: '../img/hero.jpg',
      alt: 'Djoupe_Audrey',
    },
    relatedArticles: [{ id: 'serverless' }, { id: 'ai' }],
  },
];
