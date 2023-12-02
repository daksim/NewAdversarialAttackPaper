# Latest Large Language Model Attack Papers
**update at 2023-12-02 11:28:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Locally Differentially Private Document Generation Using Zero Shot Prompting**

cs.CL

Accepted at EMNLP 2023 (Findings)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2310.16111v2) [paper-pdf](http://arxiv.org/pdf/2310.16111v2)

**Authors**: Saiteja Utpala, Sara Hooker, Pin Yu Chen

**Abstract**: Numerous studies have highlighted the privacy risks associated with pretrained large language models. In contrast, our research offers a unique perspective by demonstrating that pretrained large language models can effectively contribute to privacy preservation. We propose a locally differentially private mechanism called DP-Prompt, which leverages the power of pretrained large language models and zero-shot prompting to counter author de-anonymization attacks while minimizing the impact on downstream utility. When DP-Prompt is used with a powerful language model like ChatGPT (gpt-3.5), we observe a notable reduction in the success rate of de-anonymization attacks, showing that it surpasses existing approaches by a considerable margin despite its simpler design. For instance, in the case of the IMDB dataset, DP-Prompt (with ChatGPT) perfectly recovers the clean sentiment F1 score while achieving a 46\% reduction in author identification F1 score against static attackers and a 26\% reduction against adaptive attackers. We conduct extensive experiments across six open-source large language models, ranging up to 7 billion parameters, to analyze various effects of the privacy-utility tradeoff.



## **2. Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements**

cs.AI

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2302.09270v3) [paper-pdf](http://arxiv.org/pdf/2302.09270v3)

**Authors**: Jiawen Deng, Jiale Cheng, Hao Sun, Zhexin Zhang, Minlie Huang

**Abstract**: As generative large model capabilities advance, safety concerns become more pronounced in their outputs. To ensure the sustainable growth of the AI ecosystem, it's imperative to undertake a holistic evaluation and refinement of associated safety risks. This survey presents a framework for safety research pertaining to large models, delineating the landscape of safety risks as well as safety evaluation and improvement methods. We begin by introducing safety issues of wide concern, then delve into safety evaluation methods for large models, encompassing preference-based testing, adversarial attack approaches, issues detection, and other advanced evaluation methods. Additionally, we explore the strategies for enhancing large model safety from training to deployment, highlighting cutting-edge safety approaches for each stage in building large models. Finally, we discuss the core challenges in advancing towards more responsible AI, including the interpretability of safety mechanisms, ongoing safety issues, and robustness against malicious attacks. Through this survey, we aim to provide clear technical guidance for safety researchers and encourage further study on the safety of large models.



## **3. Leveraging a Randomized Key Matrix to Enhance the Security of Symmetric Substitution Ciphers**

cs.CR

In Proceedings of the 10th IEEE Asia-Pacific Conference on Computer  Science and Data Engineering 2023 (CSDE)

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.18085v1) [paper-pdf](http://arxiv.org/pdf/2311.18085v1)

**Authors**: Shubham Gandhi, Om Khare, Mihika Dravid, Mihika Sanghvi, Sunil Mane, Aadesh Gajaralwar, Saloni Gandhi

**Abstract**: An innovative strategy to enhance the security of symmetric substitution ciphers is presented, through the implementation of a randomized key matrix suitable for various file formats, including but not limited to binary and text files. Despite their historical relevance, symmetric substitution ciphers have been limited by vulnerabilities to cryptanalytic methods like frequency analysis and known plaintext attacks. The aim of our research is to mitigate these vulnerabilities by employing a polyalphabetic substitution strategy that incorporates a distinct randomized key matrix. This matrix plays a pivotal role in generating a unique random key, comprising characters, encompassing both uppercase and lowercase letters, numeric, and special characters, to derive the corresponding ciphertext. The effectiveness of the proposed methodology in enhancing the security of conventional substitution methods for file encryption and decryption is supported by comprehensive testing and analysis, which encompass computational speed, frequency analysis, keyspace examination, Kasiski test, entropy analysis, and the utilization of a large language model.



## **4. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2310.03684v3) [paper-pdf](http://arxiv.org/pdf/2310.03684v3)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM. Our code is publicly available at the following link: https://github.com/arobey1/smooth-llm.



## **5. Query-Relevant Images Jailbreak Large Multi-Modal Models**

cs.CV

Technique report

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17600v1) [paper-pdf](http://arxiv.org/pdf/2311.17600v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **6. Unveiling the Implicit Toxicity in Large Language Models**

cs.CL

EMNLP 2023 Main Conference

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17391v1) [paper-pdf](http://arxiv.org/pdf/2311.17391v1)

**Authors**: Jiaxin Wen, Pei Ke, Hao Sun, Zhexin Zhang, Chengfei Li, Jinfeng Bai, Minlie Huang

**Abstract**: The open-endedness of large language models (LLMs) combined with their impressive capabilities may lead to new safety issues when being exploited for malicious use. While recent studies primarily focus on probing toxic outputs that can be easily detected with existing toxicity classifiers, we show that LLMs can generate diverse implicit toxic outputs that are exceptionally difficult to detect via simply zero-shot prompting. Moreover, we propose a reinforcement learning (RL) based attacking method to further induce the implicit toxicity in LLMs. Specifically, we optimize the language model with a reward that prefers implicit toxic outputs to explicit toxic and non-toxic ones. Experiments on five widely-adopted toxicity classifiers demonstrate that the attack success rate can be significantly improved through RL fine-tuning. For instance, the RL-finetuned LLaMA-13B model achieves an attack success rate of 90.04% on BAD and 62.85% on Davinci003. Our findings suggest that LLMs pose a significant threat in generating undetectable implicit toxic outputs. We further show that fine-tuning toxicity classifiers on the annotated examples from our attacking method can effectively enhance their ability to detect LLM-generated implicit toxic language. The code is publicly available at https://github.com/thu-coai/Implicit-Toxicity.



## **7. Identifying and Mitigating Vulnerabilities in LLM-Integrated Applications**

cs.CR

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.16153v2) [paper-pdf](http://arxiv.org/pdf/2311.16153v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Boxin Wang, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: Large language models (LLMs) are increasingly deployed as the service backend for LLM-integrated applications such as code completion and AI-powered search. LLM-integrated applications serve as middleware to refine users' queries with domain-specific knowledge to better inform LLMs and enhance the responses. Despite numerous opportunities and benefits, LLM-integrated applications also introduce new attack surfaces. Understanding, minimizing, and eliminating these emerging attack surfaces is a new area of research. In this work, we consider a setup where the user and LLM interact via an LLM-integrated application in the middle. We focus on the communication rounds that begin with user's queries and end with LLM-integrated application returning responses to the queries, powered by LLMs at the service backend. For this query-response protocol, we identify potential vulnerabilities that can originate from the malicious application developer or from an outsider threat initiator that is able to control the database access, manipulate and poison data that are high-risk for the user. Successful exploits of the identified vulnerabilities result in the users receiving responses tailored to the intent of a threat initiator. We assess such threats against LLM-integrated applications empowered by OpenAI GPT-3.5 and GPT-4. Our empirical results show that the threats can effectively bypass the restrictions and moderation policies of OpenAI, resulting in users receiving responses that contain bias, toxic content, privacy risk, and disinformation. To mitigate those threats, we identify and define four key properties, namely integrity, source identification, attack detectability, and utility preservation, that need to be satisfied by a safe LLM-integrated application. Based on these properties, we develop a lightweight, threat-agnostic defense that mitigates both insider and outsider threats.



## **8. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2309.02705v2) [paper-pdf](http://arxiv.org/pdf/2309.02705v2)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial attacks, which add maliciously designed token sequences to a harmful prompt to bypass the model's safety guards. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 92% of harmful prompts and labels 94% of safe prompts correctly using the open-source language model Llama 2 as the safety filter. We further improve the filter's performance, in terms of accuracy and speed, by replacing Llama 2 with a DistilBERT safety classifier fine-tuned on safe and harmful prompts. Additionally, we propose two efficient empirical defenses: i) RandEC, a randomized version of erase-and-check that evaluates the safety filter on a small subset of the erased subsequences, and ii) GradEC, a gradient-based version that optimizes the erased tokens to remove the adversarial sequence. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.



## **9. C-ITS Environment Modeling and Attack Modeling**

cs.CR

in Korean Language, 14 Figures, 15 Pages

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.14327v2) [paper-pdf](http://arxiv.org/pdf/2311.14327v2)

**Authors**: Jaewoong Choi, Min Geun Song, Hyosun Lee, Chaeyeon Sagong, Sangbeom Park, Jaesung Lee, Jeong Do Yoo, Huy Kang Kim

**Abstract**: As technology advances, cities are evolving into smart cities, with the ability to process large amounts of data and the increasing complexity and diversification of various elements within urban areas. Among the core systems of a smart city is the Cooperative-Intelligent Transport Systems (C-ITS). C-ITS is a system where vehicles provide real-time information to drivers about surrounding traffic conditions, sudden stops, falling objects, and other accident risks through roadside base stations. It consists of road infrastructure, C-ITS centers, and vehicle terminals. However, as smart cities integrate many elements through networks and electronic control, they are susceptible to cybersecurity issues. In the case of cybersecurity problems in C-ITS, there is a significant risk of safety issues arising. This technical document aims to model the C-ITS environment and the services it provides, with the purpose of identifying the attack surface where security incidents could occur in a smart city environment. Subsequently, based on the identified attack surface, the document aims to construct attack scenarios and their respective stages. The document provides a description of the concept of C-ITS, followed by the description of the C-ITS environment model, service model, and attack scenario model defined by us.



## **10. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

cs.CL

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.11509v2) [paper-pdf](http://arxiv.org/pdf/2311.11509v2)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.



## **11. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP**

cs.CV

13 pages, 5 figures

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.16194v1) [paper-pdf](http://arxiv.org/pdf/2311.16194v1)

**Authors**: Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, Wei Liu

**Abstract**: Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.



## **12. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

cs.CL

The data and code can be found at  https://github.com/Leezekun/instruction-following-robustness-eval

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2308.10819v3) [paper-pdf](http://arxiv.org/pdf/2308.10819v3)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional proficiency in instruction-following, becoming increasingly crucial across various applications. However, this capability brings with it the risk of prompt injection attacks, where attackers inject instructions into LLMs' input to elicit undesirable actions or content. Understanding the robustness of LLMs against such attacks is vital for their safe implementation. In this work, we establish a benchmark to evaluate the robustness of instruction-following LLMs against prompt injection attacks. Our objective is to determine the extent to which LLMs can be influenced by injected instructions and their ability to differentiate between these injected and original target instructions. Through extensive experiments with leading instruction-following LLMs, we uncover significant vulnerabilities in their robustness to such attacks. Our results indicate that some models are overly tuned to follow any embedded instructions in the prompt, overly focusing on the latter parts of the prompt without fully grasping the entire context. By contrast, models with a better grasp of the context and instruction-following capabilities will potentially be more susceptible to compromise by injected instructions. This underscores the need to shift the focus from merely enhancing LLMs' instruction-following capabilities to improving their overall comprehension of prompts and discernment of instructions that are appropriate to follow. We hope our in-depth analysis offers insights into the underlying causes of these vulnerabilities, aiding in the development of future solutions. Code and data are available at https://github.com/Leezekun/instruction-following-robustness-eval



## **13. Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles**

cs.HC

10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14876v1) [paper-pdf](http://arxiv.org/pdf/2311.14876v1)

**Authors**: Sonali Singh, Faranak Abri, Akbar Siami Namin

**Abstract**: With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions.   This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.



## **14. Backdoor Activation Attack: Attack Large Language Models using Activation Steering for Safety-Alignment**

cs.CR

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.09433v2) [paper-pdf](http://arxiv.org/pdf/2311.09433v2)

**Authors**: Haoran Wang, Kai Shu

**Abstract**: To ensure AI safety, instruction-tuned Large Language Models (LLMs) are specifically trained to ensure alignment, which refers to making models behave in accordance with human intentions. While these models have demonstrated commendable results on various safety benchmarks, the vulnerability of their safety alignment has not been extensively studied. This is particularly troubling given the potential harm that LLMs can inflict. Existing attack methods on LLMs often rely on poisoned training data or the injection of malicious prompts. These approaches compromise the stealthiness and generalizability of the attacks, making them susceptible to detection. Additionally, these models often demand substantial computational resources for implementation, making them less practical for real-world applications. Inspired by recent success in modifying model behavior through steering vectors without the need for optimization, and drawing on its effectiveness in red-teaming LLMs, we conducted experiments employing activation steering to target four key aspects of LLMs: truthfulness, toxicity, bias, and harmfulness - across a varied set of attack settings. To establish a universal attack strategy applicable to diverse target alignments without depending on manual analysis, we automatically select the intervention layer based on contrastive layer search. Our experiment results show that activation attacks are highly effective and add little or no overhead to attack efficiency. Additionally, we discuss potential countermeasures against such activation attacks. Our code and data are available at https://github.com/wang2226/Backdoor-Activation-Attack Warning: this paper contains content that can be offensive or upsetting.



## **15. Universal Jailbreak Backdoors from Poisoned Human Feedback**

cs.AI

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14455v1) [paper-pdf](http://arxiv.org/pdf/2311.14455v1)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.



## **16. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation**

cs.CL

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.03348v2) [paper-pdf](http://arxiv.org/pdf/2311.03348v2)

**Authors**: Rusheb Shah, Quentin Feuillade--Montixi, Soroush Pour, Arush Tagade, Stephen Casper, Javier Rando

**Abstract**: Despite efforts to align large language models to produce harmless responses, they are still vulnerable to jailbreak prompts that elicit unrestricted behaviour. In this work, we investigate persona modulation as a black-box jailbreaking method to steer a target model to take on personalities that are willing to comply with harmful instructions. Rather than manually crafting prompts for each persona, we automate the generation of jailbreaks using a language model assistant. We demonstrate a range of harmful completions made possible by persona modulation, including detailed instructions for synthesising methamphetamine, building a bomb, and laundering money. These automated attacks achieve a harmful completion rate of 42.5% in GPT-4, which is 185 times larger than before modulation (0.23%). These prompts also transfer to Claude 2 and Vicuna with harmful completion rates of 61.0% and 35.9%, respectively. Our work reveals yet another vulnerability in commercial large language models and highlights the need for more comprehensive safeguards.



## **17. Input Reconstruction Attack against Vertical Federated Large Language Models**

cs.CL

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.07585v2) [paper-pdf](http://arxiv.org/pdf/2311.07585v2)

**Authors**: Fei Zheng

**Abstract**: Recently, large language models (LLMs) have drawn extensive attention from academia and the public, due to the advent of the ChatGPT. While LLMs show their astonishing ability in text generation for various tasks, privacy concerns limit their usage in real-life businesses. More specifically, either the user's inputs (the user sends the query to the model-hosting server) or the model (the user downloads the complete model) itself will be revealed during the usage. Vertical federated learning (VFL) is a promising solution to this kind of problem. It protects both the user's input and the knowledge of the model by splitting the model into a bottom part and a top part, which is maintained by the user and the model provider, respectively. However, in this paper, we demonstrate that in LLMs, VFL fails to protect the user input since it is simple and cheap to reconstruct the input from the intermediate embeddings. Experiments show that even with a commercial GPU, the input sentence can be reconstructed in only one second. We also discuss several possible solutions to enhance the privacy of vertical federated LLMs.



## **18. Fewer is More: Trojan Attacks on Parameter-Efficient Fine-Tuning**

cs.CL

20 pages, 6 figures

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2310.00648v3) [paper-pdf](http://arxiv.org/pdf/2310.00648v3)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance comparable to full fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we conduct a pilot study revealing that PEFT exhibits unique vulnerability to trojan attacks. Specifically, we present PETA, a novel attack that accounts for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a PLM while the lower-level objective simulates PEFT to retain the PLM's task-specific performance. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and unaffected clean accuracy, even after the victim user performs PEFT over the backdoored PLM using untainted data. Moreover, we empirically provide possible explanations for PETA's efficacy: the bilevel optimization inherently 'orthogonalizes' the backdoor and PEFT modules, thereby retaining the backdoor throughout PEFT. Based on this insight, we explore a simple defense that omits PEFT in selected layers of the backdoored PLM and unfreezes a subset of these layers' parameters, which is shown to effectively neutralize PETA.



## **19. Transfer Attacks and Defenses for Large Language Models on Coding Tasks**

cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13445v1) [paper-pdf](http://arxiv.org/pdf/2311.13445v1)

**Authors**: Chi Zhang, Zifan Wang, Ravi Mangal, Matt Fredrikson, Limin Jia, Corina Pasareanu

**Abstract**: Modern large language models (LLMs), such as ChatGPT, have demonstrated impressive capabilities for coding tasks including writing and reasoning about code. They improve upon previous neural network models of code, such as code2seq or seq2seq, that already demonstrated competitive results when performing tasks such as code summarization and identifying code vulnerabilities. However, these previous code models were shown vulnerable to adversarial examples, i.e. small syntactic perturbations that do not change the program's semantics, such as the inclusion of "dead code" through false conditions or the addition of inconsequential print statements, designed to "fool" the models. LLMs can also be vulnerable to the same adversarial perturbations but a detailed study on this concern has been lacking so far. In this paper we aim to investigate the effect of adversarial perturbations on coding tasks with LLMs. In particular, we study the transferability of adversarial examples, generated through white-box attacks on smaller code models, to LLMs. Furthermore, to make the LLMs more robust against such adversaries without incurring the cost of retraining, we propose prompt-based defenses that involve modifying the prompt to include additional information such as examples of adversarially perturbed code and explicit instructions for reversing adversarial perturbations. Our experiments show that adversarial examples obtained with a smaller code model are indeed transferable, weakening the LLMs' performance. The proposed defenses show promise in improving the model's resilience, paving the way to more robust defensive solutions for LLMs in code-related applications.



## **20. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

cs.CL

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2309.01446v3) [paper-pdf](http://arxiv.org/pdf/2309.01446v3)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.



## **21. Generating Valid and Natural Adversarial Examples with Large Language Models**

cs.CL

Submitted to the IEEE for possible publication

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11861v1) [paper-pdf](http://arxiv.org/pdf/2311.11861v1)

**Authors**: Zimu Wang, Wei Wang, Qi Chen, Qiufeng Wang, Anh Nguyen

**Abstract**: Deep learning-based natural language processing (NLP) models, particularly pre-trained language models (PLMs), have been revealed to be vulnerable to adversarial attacks. However, the adversarial examples generated by many mainstream word-level adversarial attack models are neither valid nor natural, leading to the loss of semantic maintenance, grammaticality, and human imperceptibility. Based on the exceptional capacity of language understanding and generation of large language models (LLMs), we propose LLM-Attack, which aims at generating both valid and natural adversarial examples with LLMs. The method consists of two stages: word importance ranking (which searches for the most vulnerable words) and word synonym replacement (which substitutes them with their synonyms obtained from LLMs). Experimental results on the Movie Review (MR), IMDB, and Yelp Review Polarity datasets against the baseline adversarial attack models illustrate the effectiveness of LLM-Attack, and it outperforms the baselines in human and GPT-4 evaluation by a significant margin. The model can generate adversarial examples that are typically valid and natural, with the preservation of semantic meaning, grammaticality, and human imperceptibility.



## **22. Evil Geniuses: Delving into the Safety of LLM-based Agents**

cs.CL

13 pages

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11855v1) [paper-pdf](http://arxiv.org/pdf/2311.11855v1)

**Authors**: Yu Tian, Xiao Yang, Jingyuan Zhang, Yinpeng Dong, Hang Su

**Abstract**: The rapid advancements in large language models (LLMs) have led to a resurgence in LLM-based agents, which demonstrate impressive human-like behaviors and cooperative capabilities in various interactions and strategy formulations. However, evaluating the safety of LLM-based agents remains a complex challenge. This paper elaborately conducts a series of manual jailbreak prompts along with a virtual chat-powered evil plan development team, dubbed Evil Geniuses, to thoroughly probe the safety aspects of these agents. Our investigation reveals three notable phenomena: 1) LLM-based agents exhibit reduced robustness against malicious attacks. 2) the attacked agents could provide more nuanced responses. 3) the detection of the produced improper responses is more challenging. These insights prompt us to question the effectiveness of LLM-based attacks on agents, highlighting vulnerabilities at various levels and within different role specializations within the system/agent of LLM-based agents. Extensive evaluation and discussion reveal that LLM-based agents face significant challenges in safety and yield insights for future research. Our code is available at https://github.com/T1aNS1R/Evil-Geniuses.



## **23. Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems**

cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11796v1) [paper-pdf](http://arxiv.org/pdf/2311.11796v1)

**Authors**: Guangjing Wang, Ce Zhou, Yuanda Wang, Bocheng Chen, Hanqing Guo, Qiben Yan

**Abstract**: Artificial Intelligence (AI) systems such as autonomous vehicles, facial recognition, and speech recognition systems are increasingly integrated into our daily lives. However, despite their utility, these AI systems are vulnerable to a wide range of attacks such as adversarial, backdoor, data poisoning, membership inference, model inversion, and model stealing attacks. In particular, numerous attacks are designed to target a particular model or system, yet their effects can spread to additional targets, referred to as transferable attacks. Although considerable efforts have been directed toward developing transferable attacks, a holistic understanding of the advancements in transferable attacks remains elusive. In this paper, we comprehensively explore learning-based attacks from the perspective of transferability, particularly within the context of cyber-physical security. We delve into different domains -- the image, text, graph, audio, and video domains -- to highlight the ubiquitous and pervasive nature of transferable attacks. This paper categorizes and reviews the architecture of existing attacks from various viewpoints: data, process, model, and system. We further examine the implications of transferable attacks in practical scenarios such as autonomous driving, speech recognition, and large language models (LLMs). Additionally, we outline the potential research directions to encourage efforts in exploring the landscape of transferable attacks. This survey offers a holistic understanding of the prevailing transferable attacks and their impacts across different domains.



## **24. SecureBERT and LLAMA 2 Empowered Control Area Network Intrusion Detection and Classification**

cs.CR

13 pages, 13 figures, 6 tables

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.12074v1) [paper-pdf](http://arxiv.org/pdf/2311.12074v1)

**Authors**: Xuemei Li, Huirong Fu

**Abstract**: Numerous studies have proved their effective strength in detecting Control Area Network (CAN) attacks. In the realm of understanding the human semantic space, transformer-based models have demonstrated remarkable effectiveness. Leveraging pre-trained transformers has become a common strategy in various language-related tasks, enabling these models to grasp human semantics more comprehensively. To delve into the adaptability evaluation on pre-trained models for CAN intrusion detection, we have developed two distinct models: CAN-SecureBERT and CAN-LLAMA2. Notably, our CAN-LLAMA2 model surpasses the state-of-the-art models by achieving an exceptional performance 0.999993 in terms of balanced accuracy, precision detection rate, F1 score, and a remarkably low false alarm rate of 3.10e-6. Impressively, the false alarm rate is 52 times smaller than that of the leading model, MTH-IDS (Multitiered Hybrid Intrusion Detection System). Our study underscores the promise of employing a Large Language Model as the foundational model, while incorporating adapters for other cybersecurity-related tasks and maintaining the model's inherent language-related capabilities.



## **25. A Security Risk Taxonomy for Large Language Models**

cs.CR

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.11415v1) [paper-pdf](http://arxiv.org/pdf/2311.11415v1)

**Authors**: Erik Derner, Kristina Batistič, Jan Zahálka, Robert Babuška

**Abstract**: As large language models (LLMs) permeate more and more applications, an assessment of their associated security risks becomes increasingly necessary. The potential for exploitation by malicious actors, ranging from disinformation to data breaches and reputation damage, is substantial. This paper addresses a gap in current research by focusing on the security risks posed by LLMs, which extends beyond the widely covered ethical and societal implications. Our work proposes a taxonomy of security risks along the user-model communication pipeline, explicitly focusing on prompt-based attacks on LLMs. We categorize the attacks by target and attack type within a prompt-based interaction scheme. The taxonomy is reinforced with specific attack examples to showcase the real-world impact of these risks. Through this taxonomy, we aim to inform the development of robust and secure LLM applications, enhancing their safety and trustworthiness.



## **26. FunctionMarker: Watermarking Language Datasets via Knowledge Injection**

cs.CR

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.09535v2) [paper-pdf](http://arxiv.org/pdf/2311.09535v2)

**Authors**: Shuai Li, Kejiang Chen, Kunsheng Tang, Wen Huang, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: Large Language Models (LLMs) have demonstrated superior performance in various natural language processing tasks. Meanwhile, they require extensive training data, raising concerns related to dataset copyright protection. Backdoor-based watermarking is a viable approach to protect the copyright of classification datasets. However, these methods may introduce malicious misclassification behaviors into watermarked LLMs by attackers and also affect the semantic information of the watermarked text. To address these issues, we propose FunctionMarker, a novel copyright protection method for language datasets via knowledge injection. FunctionMarker enables LLMs to learn specific knowledge through fine-tuning on watermarked datasets, and we can extract the embedded watermark by obtaining the responses of LLMs to specific knowledge-related queries. Considering watermark capacity and stealthness, we select customizable functions as specific knowledge for LLMs to learn and embed the watermark into them. Moreover, FunctionMarker can embed multi-bit watermarks while preserving the original semantic information, thereby increasing the difficulty of adaptive attacks. We take mathematical functions as an instance to evaluate the effectiveness of FunctionMarker, and experiments show that only 0.3% of watermarked text achieves a 90% watermark extraction accuracy in most cases, validating our method's effectiveness.



## **27. Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities**

cs.CR

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.16169v1) [paper-pdf](http://arxiv.org/pdf/2311.16169v1)

**Authors**: Avishree Khare, Saikat Dutta, Ziyang Li, Alaia Solko-Breslin, Rajeev Alur, Mayur Naik

**Abstract**: Security vulnerabilities in modern software are prevalent and harmful. While automated vulnerability detection tools have made promising progress, their scalability and applicability remain challenging. Recently, Large Language Models (LLMs), such as GPT-4 and CodeLlama, have demonstrated remarkable performance on code-related tasks. However, it is unknown whether such LLMs can do complex reasoning over code. In this work, we explore whether pre-trained LLMs can detect security vulnerabilities and address the limitations of existing tools. We evaluate the effectiveness of pre-trained LLMs on a set of five diverse security benchmarks spanning two languages, Java and C/C++, and including code samples from synthetic and real-world projects. We evaluate the effectiveness of LLMs in terms of their performance, explainability, and robustness.   By designing a series of effective prompting strategies, we obtain the best results on the synthetic datasets with GPT-4: F1 scores of 0.79 on OWASP, 0.86 on Juliet Java, and 0.89 on Juliet C/C++. Expectedly, the performance of LLMs drops on the more challenging real-world datasets: CVEFixes Java and CVEFixes C/C++, with GPT-4 reporting F1 scores of 0.48 and 0.62, respectively. We show that LLMs can often perform better than existing static analysis and deep learning-based vulnerability detection tools, especially for certain classes of vulnerabilities. Moreover, LLMs also often provide reliable explanations, identifying the vulnerable data flows in code. We find that fine-tuning smaller LLMs can outperform the larger LLMs on synthetic datasets but provide limited gains on real-world datasets. When subjected to adversarial attacks on code, LLMs show mild degradation, with average accuracy reduction of up to 12.67%. Finally, we share our insights and recommendations for future work on leveraging LLMs for vulnerability detection.



## **28. Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking**

cs.CL

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09827v1) [paper-pdf](http://arxiv.org/pdf/2311.09827v1)

**Authors**: Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, Muhao Chen

**Abstract**: While large language models (LLMs) have demonstrated increasing power, they have also given rise to a wide range of harmful behaviors. As representatives, jailbreak attacks can provoke harmful or unethical responses from LLMs, even after safety alignment. In this paper, we investigate a novel category of jailbreak attacks specifically designed to target the cognitive structure and processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in the face of (1) multilingual cognitive overload, (2) veiled expression, and (3) effect-to-cause reasoning. Different from previous jailbreak attacks, our proposed cognitive overload is a black-box attack with no need for knowledge of model architecture or access to model weights. Experiments conducted on AdvBench and MasterKey reveal that various LLMs, including both popular open-source model Llama 2 and the proprietary model ChatGPT, can be compromised through cognitive overload. Motivated by cognitive psychology work on managing cognitive load, we further investigate defending cognitive overload attack from two perspectives. Empirical studies show that our cognitive overload from three perspectives can jailbreak all studied LLMs successfully, while existing defense strategies can hardly mitigate the caused malicious uses effectively.



## **29. Test-time Backdoor Mitigation for Black-Box Large Language Models with Defensive Demonstrations**

cs.CL

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09763v1) [paper-pdf](http://arxiv.org/pdf/2311.09763v1)

**Authors**: Wenjie Mo, Jiashu Xu, Qin Liu, Jiongxiao Wang, Jun Yan, Chaowei Xiao, Muhao Chen

**Abstract**: Existing studies in backdoor defense have predominantly focused on the training phase, overlooking the critical aspect of testing time defense. This gap becomes particularly pronounced in the context of Large Language Models (LLMs) deployed as Web Services, which typically offer only black-box access, rendering training-time defenses impractical. To bridge this gap, our work introduces defensive demonstrations, an innovative backdoor defense strategy for blackbox large language models. Our method involves identifying the task and retrieving task-relevant demonstrations from an uncontaminated pool. These demonstrations are then combined with user queries and presented to the model during testing, without requiring any modifications/tuning to the black-box model or insights into its internal mechanisms. Defensive demonstrations are designed to counteract the adverse effects of triggers, aiming to recalibrate and correct the behavior of poisoned models during test-time evaluations. Extensive experiments show that defensive demonstrations are effective in defending both instance-level and instruction-level backdoor attacks, not only rectifying the behavior of poisoned models but also surpassing existing baselines in most scenarios.



## **30. On the Exploitability of Reinforcement Learning with Human Feedback for Large Language Models**

cs.AI

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09641v1) [paper-pdf](http://arxiv.org/pdf/2311.09641v1)

**Authors**: Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.



## **31. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

cs.CL

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09447v1) [paper-pdf](http://arxiv.org/pdf/2311.09447v1)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose an enhanced Chain of Utterances-based (CoU) prompting strategy by incorporating meticulously crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.



## **32. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

cs.CR

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09127v1) [paper-pdf](http://arxiv.org/pdf/2311.09127v1)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities in model APIs. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully steal the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2)Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking, which could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.



## **33. Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization**

cs.CL

14 pages

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09096v1) [paper-pdf](http://arxiv.org/pdf/2311.09096v1)

**Authors**: Zhexin Zhang, Junxiao Yang, Pei Ke, Minlie Huang

**Abstract**: Large Language Models (LLMs) continue to advance in their capabilities, yet this progress is accompanied by a growing array of safety risks. While significant attention has been dedicated to exploiting weaknesses in LLMs through jailbreaking attacks, there remains a paucity of exploration into defending against these attacks. We point out a pivotal factor contributing to the success of jailbreaks: the inherent conflict between the goals of being helpful and ensuring safety. To counter jailbreaking attacks, we propose to integrate goal prioritization at both training and inference stages. Implementing goal prioritization during inference substantially diminishes the Attack Success Rate (ASR) of jailbreaking attacks, reducing it from 66.4% to 2.0% for ChatGPT and from 68.2% to 19.4% for Vicuna-33B, without compromising general performance. Furthermore, integrating the concept of goal prioritization into the training phase reduces the ASR from 71.0% to 6.6% for LLama2-13B. Remarkably, even in scenarios where no jailbreaking samples are included during training, our approach slashes the ASR by half, decreasing it from 71.0% to 34.0%. Additionally, our findings reveal that while stronger LLMs face greater safety risks, they also possess a greater capacity to be steered towards defending against such attacks. We hope our work could contribute to the comprehension of jailbreaking attacks and defenses, and shed light on the relationship between LLMs' capability and safety. Our code will be available at \url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.



## **34. Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models**

cs.LG

Blog post:  https://www.harvard.edu/kempner-institute/2023/11/09/watermarking-in-the-sand/

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.04378v2) [paper-pdf](http://arxiv.org/pdf/2311.04378v2)

**Authors**: Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, Boaz Barak

**Abstract**: Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023). The same attack successfully removes the watermarks planted by all three schemes, with only minor quality degradation.



## **35. Alignment is not sufficient to prevent large language models from generating harmful information: A psychoanalytic perspective**

cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08487v1) [paper-pdf](http://arxiv.org/pdf/2311.08487v1)

**Authors**: Zi Yin, Wei Ding, Jia Liu

**Abstract**: Large Language Models (LLMs) are central to a multitude of applications but struggle with significant risks, notably in generating harmful content and biases. Drawing an analogy to the human psyche's conflict between evolutionary survival instincts and societal norm adherence elucidated in Freud's psychoanalysis theory, we argue that LLMs suffer a similar fundamental conflict, arising between their inherent desire for syntactic and semantic continuity, established during the pre-training phase, and the post-training alignment with human values. This conflict renders LLMs vulnerable to adversarial attacks, wherein intensifying the models' desire for continuity can circumvent alignment efforts, resulting in the generation of harmful information. Through a series of experiments, we first validated the existence of the desire for continuity in LLMs, and further devised a straightforward yet powerful technique, such as incomplete sentences, negative priming, and cognitive dissonance scenarios, to demonstrate that even advanced LLMs struggle to prevent the generation of harmful information. In summary, our study uncovers the root of LLMs' vulnerabilities to adversarial attacks, hereby questioning the efficacy of solely relying on sophisticated alignment methods, and further advocates for a new training idea that integrates modal concepts alongside traditional amodal concepts, aiming to endow LLMs with a more nuanced understanding of real-world contexts and ethical considerations.



## **36. LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud**

cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2309.17157v3) [paper-pdf](http://arxiv.org/pdf/2309.17157v3)

**Authors**: Mengke Zhang, Tianxing He, Tianle Wang, Lu Mi, Fatemehsadat Mireshghallah, Binyi Chen, Hao Wang, Yulia Tsvetkov

**Abstract**: In the current user-server interaction paradigm of prompted generation with large language models (LLM) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of the computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attacks from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).



## **37. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08268v1) [paper-pdf](http://arxiv.org/pdf/2311.08268v1)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on another white-box model, compromising generalization or jailbreak efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we offer detailed analysis and discussion from the perspective of prompt execution priority on the failure of LLMs' defense. We hope that our research can catalyze both the academic community and LLMs vendors towards the provision of safer and more regulated Large Language Models.



## **38. Fake Alignment: Are LLMs Really Aligned Well?**

cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.05915v2) [paper-pdf](http://arxiv.org/pdf/2311.05915v2)

**Authors**: Yixu Wang, Yan Teng, Kexin Huang, Chengqi Lyu, Songyang Zhang, Wenwei Zhang, Xingjun Ma, Yu-Gang Jiang, Yu Qiao, Yingchun Wang

**Abstract**: The growing awareness of safety concerns in large language models (LLMs) has sparked considerable interest in the evaluation of safety within current research endeavors. This study investigates an interesting issue pertaining to the evaluation of LLMs, namely the substantial discrepancy in performance between multiple-choice questions and open-ended questions. Inspired by research on jailbreak attack patterns, we argue this is caused by mismatched generalization. That is, the LLM does not have a comprehensive understanding of the complex concept of safety. Instead, it only remembers what to answer for open-ended safety questions, which makes it unable to solve other forms of safety tests. We refer to this phenomenon as fake alignment and construct a comparative benchmark to empirically verify its existence in LLMs. Such fake alignment renders previous evaluation protocols unreliable. To address this, we introduce the Fake alIgNment Evaluation (FINE) framework and two novel metrics--Consistency Score (CS) and Consistent Safety Score (CSS), which jointly assess two complementary forms of evaluation to quantify fake alignment and obtain corrected performance estimates. Applying FINE to 14 widely-used LLMs reveals several models with purported safety are poorly aligned in practice. Our work highlights potential limitations in prevailing alignment methodologies.



## **39. MART: Improving LLM Safety with Multi-round Automatic Red-Teaming**

cs.CL

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07689v1) [paper-pdf](http://arxiv.org/pdf/2311.07689v1)

**Authors**: Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao

**Abstract**: Red-teaming is a common practice for mitigating unsafe behaviors in Large Language Models (LLMs), which involves thoroughly assessing LLMs to identify potential flaws and addressing them with responsible and accurate responses. While effective, manual red-teaming is costly, and existing automatic red-teaming typically discovers safety risks without addressing them. In this paper, we propose a Multi-round Automatic Red-Teaming (MART) method, which incorporates both automatic adversarial prompt writing and safe response generation, significantly increasing red-teaming scalability and the safety of the target LLM. Specifically, an adversarial LLM and a target LLM interplay with each other in an iterative manner, where the adversarial LLM aims to generate challenging prompts that elicit unsafe responses from the target LLM, while the target LLM is fine-tuned with safety aligned data on these adversarial prompts. In each round, the adversarial LLM crafts better attacks on the updated target LLM, while the target LLM also improves itself through safety fine-tuning. On adversarial prompt benchmarks, the violation rate of an LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART, achieving comparable performance to LLMs with extensive adversarial prompt writing. Notably, model helpfulness on non-adversarial prompts remains stable throughout iterations, indicating the target LLM maintains strong performance on instruction following.



## **40. Summon a Demon and Bind it: A Grounded Theory of LLM Red Teaming in the Wild**

cs.CL

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.06237v2) [paper-pdf](http://arxiv.org/pdf/2311.06237v2)

**Authors**: Nanna Inie, Jonathan Stray, Leon Derczynski

**Abstract**: Engaging in the deliberate generation of abnormal outputs from large language models (LLMs) by attacking them is a novel human activity. This paper presents a thorough exposition of how and why people perform such attacks. Using a formal qualitative methodology, we interviewed dozens of practitioners from a broad range of backgrounds, all contributors to this novel work of attempting to cause LLMs to fail. We relate and connect this activity between its practitioners' motivations and goals; the strategies and techniques they deploy; and the crucial role the community plays. As a result, this paper presents a grounded theory of how and why people attack large language models: LLM red teaming in the wild.



## **41. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

cs.CL

Under Review

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2310.14303v2) [paper-pdf](http://arxiv.org/pdf/2310.14303v2)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.



## **42. Removing RLHF Protections in GPT-4 via Fine-Tuning**

cs.CL

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.05553v2) [paper-pdf](http://arxiv.org/pdf/2311.05553v2)

**Authors**: Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, Daniel Kang

**Abstract**: As large language models (LLMs) have increased in their capabilities, so does their potential for dual use. To reduce harmful outputs, produces and vendors of LLMs have used reinforcement learning with human feedback (RLHF). In tandem, LLM vendors have been increasingly enabling fine-tuning of their most powerful models. However, concurrent work has shown that fine-tuning can remove RLHF protections. We may expect that the most powerful models currently available (GPT-4) are less susceptible to fine-tuning attacks.   In this work, we show the contrary: fine-tuning allows attackers to remove RLHF protections with as few as 340 examples and a 95% success rate. These training examples can be automatically generated with weaker models. We further show that removing RLHF protections does not decrease usefulness on non-censored outputs, providing evidence that our fine-tuning strategy does not decrease usefulness despite using weaker models to generate training data. Our results show the need for further research on protections on LLMs.



## **43. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06062v1) [paper-pdf](http://arxiv.org/pdf/2311.06062v1)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.



## **44. Watermarking Vision-Language Pre-trained Models for Multi-modal Embedding as a Service**

cs.CR

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.05863v1) [paper-pdf](http://arxiv.org/pdf/2311.05863v1)

**Authors**: Yuanmin Tang, Jing Yu, Keke Gai, Xiangyan Qu, Yue Hu, Gang Xiong, Qi Wu

**Abstract**: Recent advances in vision-language pre-trained models (VLPs) have significantly increased visual understanding and cross-modal analysis capabilities. Companies have emerged to provide multi-modal Embedding as a Service (EaaS) based on VLPs (e.g., CLIP-based VLPs), which cost a large amount of training data and resources for high-performance service. However, existing studies indicate that EaaS is vulnerable to model extraction attacks that induce great loss for the owners of VLPs. Protecting the intellectual property and commercial ownership of VLPs is increasingly crucial yet challenging. A major solution of watermarking model for EaaS implants a backdoor in the model by inserting verifiable trigger embeddings into texts, but it is only applicable for large language models and is unrealistic due to data and model privacy. In this paper, we propose a safe and robust backdoor-based embedding watermarking method for VLPs called VLPMarker. VLPMarker utilizes embedding orthogonal transformation to effectively inject triggers into the VLPs without interfering with the model parameters, which achieves high-quality copyright verification and minimal impact on model performance. To enhance the watermark robustness, we further propose a collaborative copyright verification strategy based on both backdoor trigger and embedding distribution, enhancing resilience against various attacks. We increase the watermark practicality via an out-of-distribution trigger selection approach, removing access to the model training data and thus making it possible for many real-world scenarios. Our extensive experiments on various datasets indicate that the proposed watermarking approach is effective and safe for verifying the copyright of VLPs for multi-modal EaaS and robust against model extraction attacks. Our code is available at https://github.com/Pter61/vlpmarker.



## **45. FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts**

cs.CR

Technical Report

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05608v1) [paper-pdf](http://arxiv.org/pdf/2311.05608v1)

**Authors**: Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang

**Abstract**: Large vision-language models (VLMs) like GPT-4V represent an unprecedented revolution in the field of artificial intelligence (AI). Compared to single-modal large language models (LLMs), VLMs possess more versatile capabilities by incorporating additional modalities (e.g., images). Meanwhile, there's a rising enthusiasm in the AI community to develop open-source VLMs, such as LLaVA and MiniGPT4, which, however, have not undergone rigorous safety assessment. In this paper, to demonstrate that more modalities lead to unforeseen AI safety issues, we propose FigStep, a novel jailbreaking framework against VLMs. FigStep feeds harmful instructions into VLMs through the image channel and then uses benign text prompts to induce VLMs to output contents that violate common AI safety policies. Our experimental results show that FigStep can achieve an average attack success rate of 94.8% across 2 families of popular open-source VLMs, LLaVA and MiniGPT4 (a total of 5 VLMs). Moreover, we demonstrate that the methodology of FigStep can even jailbreak GPT-4V, which already leverages several system-level mechanisms to filter harmful queries. Above all, our experimental results reveal that VLMs are vulnerable to jailbreaking attacks, which highlights the necessity of novel safety alignments between visual and textual modalities.



## **46. Backdoor Attacks and Countermeasures in Natural Language Processing Models: A Comprehensive Security Review**

cs.CR

21 pages, 4 figures

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2309.06055v4) [paper-pdf](http://arxiv.org/pdf/2309.06055v4)

**Authors**: Pengzhou Cheng, Zongru Wu, Wei Du, Haodong Zhao, Wei Lu, Gongshen Liu

**Abstract**: Applicating third-party data and models has become a new paradigm for language modeling in NLP, which also introduces some potential security vulnerabilities because attackers can manipulate the training process and data source. In this case, backdoor attacks can induce the model to exhibit expected behaviors through specific triggers and have little inferior influence on primitive tasks. Hence, it could have dire consequences, especially considering that the backdoor attack surfaces are broad.   However, there is still no systematic and comprehensive review to reflect the security challenges, attacker's capabilities, and purposes according to the attack surface. Moreover, there is a shortage of analysis and comparison of the diverse emerging backdoor countermeasures in this context. In this paper, we conduct a timely review of backdoor attacks and countermeasures to sound the red alarm for the NLP security community. According to the affected stage of the machine learning pipeline, the attack surfaces are recognized to be wide and then formalized into three categorizations: attacking pre-trained model with fine-tuning (APMF) or parameter-efficient tuning (APMP), and attacking final model with training (AFMT). Thus, attacks under each categorization are combed. The countermeasures are categorized into two general classes: sample inspection and model inspection. Overall, the research on the defense side is far behind the attack side, and there is no single defense that can prevent all types of backdoor attacks. An attacker can intelligently bypass existing defenses with a more invisible attack. Drawing the insights from the systematic review, we also present crucial areas for future research on the backdoor, such as empirical security evaluations on large language models, and in particular, more efficient and practical countermeasures are solicited.



## **47. Unveiling Safety Vulnerabilities of Large Language Models**

cs.CL

To be published in GEM workshop. Conference on Empirical Methods in  Natural Language Processing (EMNLP). 2023

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.04124v1) [paper-pdf](http://arxiv.org/pdf/2311.04124v1)

**Authors**: George Kour, Marcel Zalmanovici, Naama Zwerdling, Esther Goldbraich, Ora Nova Fandina, Ateret Anaby-Tavor, Orna Raz, Eitan Farchi

**Abstract**: As large language models become more prevalent, their possible harmful or inappropriate responses are a cause for concern. This paper introduces a unique dataset containing adversarial examples in the form of questions, which we call AttaQ, designed to provoke such harmful or inappropriate responses. We assess the efficacy of our dataset by analyzing the vulnerabilities of various models when subjected to it. Additionally, we introduce a novel automatic approach for identifying and naming vulnerable semantic regions - input semantic areas for which the model is likely to produce harmful outputs. This is achieved through the application of specialized clustering techniques that consider both the semantic similarity of the input attacks and the harmfulness of the model's responses. Automatically identifying vulnerable semantic regions enhances the evaluation of model weaknesses, facilitating targeted improvements to its safety mechanisms and overall reliability.



## **48. Detecting Language Model Attacks with Perplexity**

cs.CL

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2308.14132v3) [paper-pdf](http://arxiv.org/pdf/2308.14132v3)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, exploiting adversarial suffixes to deceive models into generating perilous responses. Such jailbreaks can trick LLMs into providing intricate instructions to a malicious user for creating explosives, orchestrating a bank heist, or facilitating the creation of offensive content. By evaluating the perplexity of queries with adversarial suffixes using an open-source LLM (GPT-2), we found that they have exceedingly high perplexity values. As we explored a broad range of regular (non-adversarial) prompt varieties, we concluded that false positives are a significant challenge for plain perplexity filtering. A Light-GBM trained on perplexity and token length resolved the false positives and correctly detected most adversarial attacks in the test set.



## **49. Competence-Based Analysis of Language Models**

cs.CL

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2303.00333v3) [paper-pdf](http://arxiv.org/pdf/2303.00333v3)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent success of large, pretrained neural language models (LLMs) on a variety of prompting tasks, these models can be alarmingly brittle to small changes in inputs or application contexts. To better understand such behavior and motivate the design of more robust LLMs, we provide a causal formulation of linguistic competence in the context of LLMs and propose a general framework to study and measure LLM competence. Our framework, CALM (Competence-based Analysis of Language Models), establishes the first quantitative measure of LLM competence, which we study by damaging models' internal representations of various linguistic properties in the course of performing various tasks using causal probing and evaluating models' alignment under these interventions with a given causal model. We also develop a novel approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than existing techniques. We carry out a case study of CALM using these interventions to analyze BERT and RoBERTa's competence across a variety of lexical inference tasks, showing that the CALM framework and competence metric can be valuable tools for explaining and predicting their behavior across these tasks.



## **50. Vulnerabilities in AI Code Generators: Exploring Targeted Data Poisoning Attacks**

cs.CR

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2308.04451v2) [paper-pdf](http://arxiv.org/pdf/2308.04451v2)

**Authors**: Domenico Cotroneo, Cristina Improta, Pietro Liguori, Roberto Natella

**Abstract**: AI-based code generators have become pivotal in assisting developers in writing software starting from natural language (NL). However, they are trained on large amounts of data, often collected from unsanitized online sources (e.g., GitHub, HuggingFace). As a consequence, AI models become an easy target for data poisoning, i.e., an attack that injects malicious samples into the training data to generate vulnerable code. To address this threat, we investigate the security of AI code generators by devising a targeted data poisoning strategy. We poison the training data by injecting increasing amounts of code containing security vulnerabilities and assess the attack's success on different state-of-the-art models for code generation. Our study shows that AI code generators are vulnerable to even a small amount of poison. Notably, the attack success strongly depends on the model architecture and poisoning rate, whereas it is not influenced by the type of vulnerabilities. Moreover, since the attack does not impact the correctness of code generated by pre-trained models, it is hard to detect. Lastly, our work offers practical insights into understanding and potentially mitigating this threat.



