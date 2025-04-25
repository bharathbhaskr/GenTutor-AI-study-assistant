# Prompt engineering

## History

In 2018, researchers first proposed that all previously separate tasks in natural language processing (NLP) could be cast as a question-answering problem over a context. In addition, they trained a first single, joint, multi-task model that would answer any task-related question like "What is the sentiment" or "Translate this sentence to German" or "Who is the president?"
The AI boom saw an increase in the amount of "prompting technique" to get the model to output the desired outcome and avoid nonsensical output, a process characterized by trial-and-error. After the release of ChatGPT in 2022, prompt engineering was soon seen as an important business skill, albeit one with an uncertain economic future.
A repository for prompts reported that over 2,000 public prompts for around 170 datasets were available in February 2022. In 2022, the chain-of-thought prompting technique was proposed by Google researchers. In 2023, several text-to-text and text-to-image prompt databases were made publicly available. The Personalized Image-Prompt (PIP) dataset, a generated image-text dataset that has been categorized by 3,115 users, has also been made available publicly in 2024.

## Text-to-text → Chain-of-thought

According to Google Research, chain-of-thought (CoT) prompting is a technique that allows large language models (LLMs) to solve a problem as a series of intermediate steps before giving a final answer. In 2022, Google Brain reported that chain-of-thought prompting improves reasoning ability by inducing the model to answer a multi-step problem with steps of reasoning that mimic a train of thought. Chain-of-thought techniques were developed to help LLMs handle multi-step reasoning tasks, such as arithmetic or commonsense reasoning questions.
For example, given the question, "Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?", Google claims that a CoT prompt might induce the LLM to answer "A: The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9." When applied to PaLM, a 540 billion parameter language model, according to Google, CoT prompting significantly aided the model, allowing it to perform comparably with task-specific fine-tuned models on several tasks, achieving state-of-the-art results at the time on the GSM8K mathematical reasoning benchmark. It is possible to fine-tune models on CoT reasoning datasets to enhance this capability further and stimulate better interpretability.
An example of a CoT prompting:

   Q: {question}
   A: Let's think step by step.

As originally proposed by Google, each CoT prompt included a few Q&A examples. This made it a few-shot prompting technique. However, according to researchers at Google and the University of Tokyo, simply appending the words "Let's think step-by-step", has also proven effective, which makes CoT a zero-shot prompting technique. OpenAI claims that this prompt allows for better scaling as a user no longer needs to formulate many specific CoT Q&A examples.

## Text-to-text → In-context learning

In-context learning, refers to a model's ability to temporarily learn from prompts. For example, a prompt may include a few examples for a model to learn from, such as asking the model to complete "maison → house, chat → cat, chien →" (the expected response being dog), an approach called few-shot learning.
In-context learning is an emergent ability of large language models. It is an emergent property of model scale, meaning that breaks in downstream scaling laws occur, leading to its efficacy increasing at a different rate in larger models than in smaller models. Unlike training and fine-tuning, which produce lasting changes, in-context learning is temporary. Training models to perform in-context learning can be viewed as a form of meta-learning, or "learning to learn".

## Text-to-text → Self-consistency decoding

Self-consistency decoding performs several chain-of-thought rollouts, then selects the most commonly reached conclusion out of all the rollouts. If the rollouts disagree by a lot, a human can be queried for the correct chain of thought.

## Text-to-text → Tree-of-thought

Tree-of-thought prompting generalizes chain-of-thought by prompting the model to generate one or more "possible next steps", and then running the model on each of the possible next steps by breadth-first, beam, or some other method of tree search. The LLM has additional modules that can converse the history of the problem-solving process to the LLM, which allows the system to 'backtrack steps' the problem-solving process.

## Text-to-text → Prompting to disclose uncertainty

By default, the output of language models may not contain estimates of uncertainty. The model may output text that appears confident, though the underlying token predictions have low likelihood scores. Large language models like GPT-4 can have accurately calibrated likelihood scores in their token predictions, and so the model output uncertainty can be directly estimated by reading out the token prediction likelihood scores.

## Text-to-text → Prompting to estimate model sensitivity

Research consistently demonstrates that LLMs are highly sensitive to subtle variations in prompt formatting, structure, and linguistic properties. Some studies have shown up to 76 accuracy points across formatting changes in few-shot settings. Linguistic features significantly influence prompt effectiveness—such as morphology, syntax, and lexico-semantic changes—which meaningfully enhance task performance across a variety of tasks. Clausal syntax, for example, improves consistency and reduces uncertainty in knowledge retrieval. This sensitivity persists even with larger model sizes, additional few-shot examples, or instruction tuning.
To address sensitivity of models and make them more robust, several methods have been proposed. FormatSpread facilitates systematic analysis by evaluating a range of plausible prompt formats, offering a more comprehensive performance interval. Similarly, PromptEval estimates performance distributions across diverse prompts, enabling robust metrics such as performance quantiles and accurate evaluations under constrained budgets.

## Text-to-text → Automatic prompt generation → Retrieval-augmented generation

Retrieval-augmented generation (RAG) is a technique that enables generative artificial intelligence (Gen AI) models to retrieve and incorporate new information. It modifies interactions with a large language model (LLM) so that the model responds to user queries with reference to a specified set of documents, using this information to supplement information from its pre-existing training data. This allows LLMs to use domain-specific and/or updated information.
RAG improves large language models (LLMs) by incorporating information retrieval before generating responses. Unlike traditional LLMs that rely on static training data, RAG pulls relevant text from databases, uploaded documents, or web sources. According to Ars Technica, "RAG is a way of improving LLM performance, in essence by blending the LLM process with a web search or other document look-up process to help LLMs stick to the facts." This method helps reduce AI hallucinations, which have led to real-world issues like chatbots inventing policies or lawyers citing nonexistent legal cases. By dynamically retrieving information, RAG enables AI to provide more accurate responses without frequent retraining.

## Text-to-text → Automatic prompt generation → Graph retrieval-augmented generation

GraphRAG (coined by Microsoft Research) is a technique that extends RAG with the use of a knowledge graph (usually, LLM-generated) to allow the model to connect disparate pieces of information, synthesize insights, and holistically understand summarized semantic concepts over large data collections. It was shown to be effective on datasets like the Violent Incident Information from News Articles (VIINA).
Earlier work showed the effectiveness of using a knowledge graph for question answering using text-to-query generation. These techniques can be combined to search across both unstructured and structured data, providing expanded context, and improved ranking.

## Text-to-text → Automatic prompt generation → Using language models to generate prompts

Large language models (LLM) themselves can be used to compose prompts for large language models. The automatic prompt engineer algorithm uses one LLM to beam search over prompts for another LLM:

There are two LLMs. One is the target LLM, and another is the prompting LLM.
Prompting LLM is presented with example input-output pairs, and asked to generate instructions that could have caused a model following the instructions to generate the outputs, given the inputs.
Each of the generated instructions is used to prompt the target LLM, followed by each of the inputs. The log-probabilities of the outputs are computed and added. This is the score of the instruction.
The highest-scored instructions are given to the prompting LLM for further variations.
Repeat until some stopping criteria is reached, then output the highest-scored instructions.
CoT examples can be generated by LLM themselves. In "auto-CoT", a library of questions are converted to vectors by a model such as BERT. The question vectors are clustered. Questions nearest to the centroids of each cluster are selected. An LLM does zero-shot CoT on each question. The resulting CoT examples are added to the dataset. When prompted with a new question, CoT examples to the nearest questions can be retrieved and added to the prompt.

## Text-to-image → Prompt formats

A text-to-image prompt commonly includes a description of the subject of the art, the desired medium (such as digital painting or photography), style (such as hyperrealistic or pop-art), lighting (such as rim lighting or crepuscular rays), color, and texture. Word order also affects the output of a text-to-image prompt. Words closer to the start of a prompt may be emphasized more heavily.
The Midjourney documentation encourages short, descriptive prompts: instead of "Show me a picture of lots of blooming California poppies, make them bright, vibrant orange, and draw them in an illustrated style with colored pencils", an effective prompt might be "Bright orange California poppies drawn with colored pencils".

## Text-to-image → Artist styles

Some text-to-image models are capable of imitating the style of particular artists by name. For example, the phrase in the style of Greg Rutkowski has been used in Stable Diffusion and Midjourney prompts to generate images in the distinctive style of Polish digital artist Greg Rutkowski. Famous artists such as Vincent van Gogh and Salvador Dalí have also been used for styling and testing.

## Non-text prompts → Textual inversion and embeddings

For text-to-image models, textual inversion performs an optimization process to create a new word embedding based on a set of example images. This embedding vector acts as a "pseudo-word" which can be included in a prompt to express the content or style of the examples.

## Non-text prompts → Image prompting

In 2023, Meta's AI research released Segment Anything, a computer vision model that can perform image segmentation by prompting. As an alternative to text prompts, Segment Anything can accept bounding boxes, segmentation masks, and foreground/background points.

## Non-text prompts → Using gradient descent to search for prompts

In "prefix-tuning", "prompt tuning", or "soft prompting", floating-point-valued vectors are searched directly by gradient descent to maximize the log-likelihood on outputs.
Formally, let 
  
    
      
        
          E
        
        =
        {
        
          
            e
            
              1
            
          
        
        ,
        …
        ,
        
          
            e
            
              k
            
          
        
        }
      
    
    {\displaystyle \mathbf {E} =\{\mathbf {e_{1}} ,\dots ,\mathbf {e_{k}} \}}
  
 be a set of soft prompt tokens (tunable embeddings), while 
  
    
      
        
          X
        
        =
        {
        
          
            x
            
              1
            
          
        
        ,
        …
        ,
        
          
            x
            
              m
            
          
        
        }
      
    
    {\displaystyle \mathbf {X} =\{\mathbf {x_{1}} ,\dots ,\mathbf {x_{m}} \}}
  
 and 
  
    
      
        
          Y
        
        =
        {
        
          
            y
            
              1
            
          
        
        ,
        …
        ,
        
          
            y
            
              n
            
          
        
        }
      
    
    {\displaystyle \mathbf {Y} =\{\mathbf {y_{1}} ,\dots ,\mathbf {y_{n}} \}}
  
 be the token embeddings of the input and output respectively. During training, the tunable embeddings, input, and output tokens are concatenated into a single sequence 
  
    
      
        
          concat
        
        (
        
          E
        
        ;
        
          X
        
        ;
        
          Y
        
        )
      
    
    {\displaystyle {\text{concat}}(\mathbf {E} ;\mathbf {X} ;\mathbf {Y} )}
  
, and fed to the LLMs. The losses are computed over the 
  
    
      
        
          Y
        
      
    
    {\displaystyle \mathbf {Y} }
  
 tokens; the gradients are backpropagated to prompt-specific parameters: in prefix-tuning, they are parameters associated with the prompt tokens at each layer; in prompt tuning, they are merely the soft tokens added to the vocabulary.
More formally, this is prompt tuning. Let an LLM be written as 
  
    
      
        L
        L
        M
        (
        X
        )
        =
        F
        (
        E
        (
        X
        )
        )
      
    
    {\displaystyle LLM(X)=F(E(X))}
  
, where 
  
    
      
        X
      
    
    {\displaystyle X}
  
 is a sequence of linguistic tokens, 
  
    
      
        E
      
    
    {\displaystyle E}
  
 is the token-to-vector function, and 
  
    
      
        F
      
    
    {\displaystyle F}
  
 is the rest of the model. In prefix-tuning, one provides a set of input-output pairs 
  
    
      
        {
        (
        
          X
          
            i
          
        
        ,
        
          Y
          
            i
          
        
        )
        
          }
          
            i
          
        
      
    
    {\displaystyle \{(X^{i},Y^{i})\}_{i}}
  
, and then use gradient descent to search for 
  
    
      
        arg
        ⁡
        
          max
          
            
              
                Z
                ~
              
            
          
        
        
          ∑
          
            i
          
        
        log
        ⁡
        P
        r
        [
        
          Y
          
            i
          
        
        
          |
        
        
          
            
              Z
              ~
            
          
        
        ∗
        E
        (
        
          X
          
            i
          
        
        )
        ]
      
    
    {\displaystyle \arg \max _{\tilde {Z}}\sum _{i}\log Pr[Y^{i}|{\tilde {Z}}\ast E(X^{i})]}
  
. In words, 
  
    
      
        log
        ⁡
        P
        r
        [
        
          Y
          
            i
          
        
        
          |
        
        
          
            
              Z
              ~
            
          
        
        ∗
        E
        (
        
          X
          
            i
          
        
        )
        ]
      
    
    {\displaystyle \log Pr[Y^{i}|{\tilde {Z}}\ast E(X^{i})]}
  
 is the log-likelihood of outputting 
  
    
      
        
          Y
          
            i
          
        
      
    
    {\displaystyle Y^{i}}
  
, if the model first encodes the input 
  
    
      
        
          X
          
            i
          
        
      
    
    {\displaystyle X^{i}}
  
 into the vector 
  
    
      
        E
        (
        
          X
          
            i
          
        
        )
      
    
    {\displaystyle E(X^{i})}
  
, then prepend the vector with the "prefix vector" 
  
    
      
        
          
            
              Z
              ~
            
          
        
      
    
    {\displaystyle {\tilde {Z}}}
  
, then apply 
  
    
      
        F
      
    
    {\displaystyle F}
  
.
For prefix tuning, it is similar, but the "prefix vector" 
  
    
      
        
          
            
              Z
              ~
            
          
        
      
    
    {\displaystyle {\tilde {Z}}}
  
 is pre-appended to the hidden states in every layer of the model.
An earlier result uses the same idea of gradient descent search, but is designed for masked language models like BERT, and searches only over token sequences, rather than numerical vectors. Formally, it searches for 
  
    
      
        arg
        ⁡
        
          max
          
            
              
                X
                ~
              
            
          
        
        
          ∑
          
            i
          
        
        log
        ⁡
        P
        r
        [
        
          Y
          
            i
          
        
        
          |
        
        
          
            
              X
              ~
            
          
        
        ∗
        
          X
          
            i
          
        
        ]
      
    
    {\displaystyle \arg \max _{\tilde {X}}\sum _{i}\log Pr[Y^{i}|{\tilde {X}}\ast X^{i}]}
  
 where 
  
    
      
        
          
            
              X
              ~
            
          
        
      
    
    {\displaystyle {\tilde {X}}}
  
 is ranges over token sequences of a specified length.

## Prompt injection

Prompt injection is a cybersecurity exploit in which adversaries craft inputs that appear legitimate but are designed to cause unintended behavior in machine learning models, particularly large language models (LLMs). This attack takes advantage of the model's inability to distinguish between developer-defined prompts and user inputs, allowing adversaries to bypass safeguards and influence model behaviour. While LLMs are designed to follow trusted instructions, they can be manipulated into carrying out unintended responses through carefully crafted inputs.

