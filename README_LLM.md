# Latest Large Language Model Attack Papers
**update at 2023-10-25 10:02:49**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Self-Guard: Empower the LLM to Safeguard Itself**

cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15851v1) [paper-pdf](http://arxiv.org/pdf/2310.15851v1)

**Authors**: Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong

**Abstract**: The jailbreak attack can bypass the safety measures of a Large Language Model (LLM), generating harmful content. This misuse of LLM has led to negative societal consequences. Currently, there are two main approaches to address jailbreak attacks: safety training and safeguards. Safety training focuses on further training LLM to enhance its safety. On the other hand, safeguards involve implementing external models or filters to prevent harmful outputs. However, safety training has constraints in its ability to adapt to new attack types and often leads to a drop in model performance. Safeguards have proven to be of limited help. To tackle these issues, we propose a novel approach called Self-Guard, which combines the strengths of both safety methods. Self-Guard includes two stages. In the first stage, we enhance the model's ability to assess harmful content, and in the second stage, we instruct the model to consistently perform harmful content detection on its own responses. The experiment has demonstrated that Self-Guard is robust against jailbreak attacks. In the bad case analysis, we find that LLM occasionally provides harmless responses to harmful queries. Additionally, we evaluated the general capabilities of the LLM before and after safety training, providing evidence that Self-Guard does not result in the LLM's performance degradation. In sensitivity tests, Self-Guard not only avoids inducing over-sensitivity in LLM but also can even mitigate this issue.



## **2. A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions**

cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14724v2) [paper-pdf](http://arxiv.org/pdf/2310.14724v2)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, zero-shot methods, fine-turning LMs methods, adversarial learning methods, LLMs as detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, and data ambiguity. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **3. A Survey on Detection of LLMs-Generated Content**

cs.CL

We will keep updating at  https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15654v1) [paper-pdf](http://arxiv.org/pdf/2310.15654v1)

**Authors**: Xianjun Yang, Liangming Pan, Xuandong Zhao, Haifeng Chen, Linda Petzold, William Yang Wang, Wei Cheng

**Abstract**: The burgeoning capabilities of advanced large language models (LLMs) such as ChatGPT have led to an increase in synthetic content generation with implications across a variety of sectors, including media, cybersecurity, public discourse, and education. As such, the ability to detect LLMs-generated content has become of paramount importance. We aim to provide a detailed overview of existing detection strategies and benchmarks, scrutinizing their differences and identifying key challenges and prospects in the field, advocating for more adaptable and robust models to enhance detection accuracy. We also posit the necessity for a multi-faceted approach to defend against various attacks to counter the rapidly advancing capabilities of LLMs. To the best of our knowledge, this work is the first comprehensive survey on the detection in the era of LLMs. We hope it will provide a broad understanding of the current landscape of LLMs-generated content detection, offering a guiding reference for researchers and practitioners striving to uphold the integrity of digital information in an era increasingly dominated by synthetic content. The relevant papers are summarized and will be consistently updated at https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git.



## **4. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2308.07308v3) [paper-pdf](http://arxiv.org/pdf/2308.07308v3)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2.



## **5. PrivInfer: Privacy-Preserving Inference for Black-box Large Language Model**

cs.CR

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.12214v3) [paper-pdf](http://arxiv.org/pdf/2310.12214v3)

**Authors**: Meng Tong, Kejiang Chen, Yuang Qi, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: Large language models (LLMs), such as ChatGPT, have simplified text generation tasks, yet their inherent privacy risks are increasingly garnering attention. Existing solutions for privacy-preserving inference face significant challenges in practical deployment and implementation. In this paper, we propose PrivInfer, the first practical framework for privacy-preserving inference. It comprises two modules specifically designed for black-box LLMs in text generation. The perturbation module, employing differential privacy, generates perturbed prompts, thus enabling privacy-preserving inference with black-box LLMs. The restoration module extracts coherent and meaningful responses from obtained perturbed results, thus ensuring the accomplishment of the text generation tasks. Additionally, to enhance privacy and utility further, we develop RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of PrivInfer. This mechanism is specifically tailored for LLMs and utilizes random adjacency in text perturbations. Experimental results indicate that PrivInfer is comparable to GPT-4 in text generation quality, and RANTEXT outperforms the current leading scheme in privacy protection, even under its adaptive attack, our proposed GPT inference attack.



## **6. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

cs.CR

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15469v1) [paper-pdf](http://arxiv.org/pdf/2310.15469v1)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, XiaoFeng Wang, Haixu Tang

**Abstract**: The era post-2018 marked the advent of Large Language Models (LLMs), with innovations such as OpenAI's ChatGPT showcasing prodigious linguistic prowess. As the industry galloped toward augmenting model parameters and capitalizing on vast swaths of human language data, security and privacy challenges also emerged. Foremost among these is the potential inadvertent accrual of Personal Identifiable Information (PII) during web-based data acquisition, posing risks of unintended PII disclosure. While strategies like RLHF during training and Catastrophic Forgetting have been marshaled to control the risk of privacy infringements, recent advancements in LLMs, epitomized by OpenAI's fine-tuning interface for GPT-3.5, have reignited concerns. One may ask: can the fine-tuning of LLMs precipitate the leakage of personal information embedded within training datasets? This paper reports the first endeavor to seek the answer to the question, particularly our discovery of a new LLM exploitation avenue, called the Janus attack. In the attack, one can construct a PII association task, whereby an LLM is fine-tuned using a minuscule PII dataset, to potentially reinstate and reveal concealed PIIs. Our findings indicate that, with a trivial fine-tuning outlay, LLMs such as GPT-3.5 can transition from being impermeable to PII extraction to a state where they divulge a substantial proportion of concealed PII. This research, through its deep dive into the Janus attack vector, underscores the imperative of navigating the intricate interplay between LLM utility and privacy preservation.



## **7. AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15140v1) [paper-pdf](http://arxiv.org/pdf/2310.15140v1)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent work suggests that patching LLMs against these attacks is possible: manual jailbreak attacks are human-readable but often limited and public, making them easy to block; adversarial attacks generate gibberish prompts that can be detected using perplexity-based filters. In this paper, we show that these solutions may be too optimistic. We propose an interpretable adversarial attack, \texttt{AutoDAN}, that combines the strengths of both types of attacks. It automatically generates attack prompts that bypass perplexity-based filters while maintaining a high attack success rate like manual jailbreak attacks. These prompts are interpretable and diverse, exhibiting strategies commonly used in manual jailbreak attacks, and transfer better than their non-readable counterparts when using limited training data or a single proxy model. We also customize \texttt{AutoDAN}'s objective to leak system prompts, another jailbreak application not addressed in the adversarial attack literature. %, demonstrating the versatility of the approach. We can also customize the objective of \texttt{AutoDAN} to leak system prompts, beyond the ability to elicit harmful content from the model, demonstrating the versatility of the approach. Our work provides a new way to red-team LLMs and to understand the mechanism of jailbreak attacks.



## **8. Did the Neurons Read your Book? Document-level Membership Inference for Large Language Models**

cs.CL

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15007v1) [paper-pdf](http://arxiv.org/pdf/2310.15007v1)

**Authors**: Matthieu Meeus, Shubham Jain, Marek Rei, Yves-Alexandre de Montjoye

**Abstract**: With large language models (LLMs) poised to become embedded in our daily lives, questions are starting to be raised about the dataset(s) they learned from. These questions range from potential bias or misinformation LLMs could retain from their training data to questions of copyright and fair use of human-generated text. However, while these questions emerge, developers of the recent state-of-the-art LLMs become increasingly reluctant to disclose details on their training corpus. We here introduce the task of document-level membership inference for real-world LLMs, i.e. inferring whether the LLM has seen a given document during training or not. First, we propose a procedure for the development and evaluation of document-level membership inference for LLMs by leveraging commonly used data sources for training and the model release date. We then propose a practical, black-box method to predict document-level membership and instantiate it on OpenLLaMA-7B with both books and academic papers. We show our methodology to perform very well, reaching an impressive AUC of 0.856 for books and 0.678 for papers. We then show our approach to outperform the sentence-level membership inference attacks used in the privacy literature for the document-level membership task. We finally evaluate whether smaller models might be less sensitive to document-level inference and show OpenLLaMA-3B to be approximately as sensitive as OpenLLaMA-7B to our approach. Taken together, our results show that accurate document-level membership can be inferred for LLMs, increasing the transparency of technology poised to change our lives.



## **9. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2306.06815v2) [paper-pdf](http://arxiv.org/pdf/2306.06815v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.



## **10. MoPe: Model Perturbation-based Privacy Attacks on Language Models**

cs.LG

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14369v1) [paper-pdf](http://arxiv.org/pdf/2310.14369v1)

**Authors**: Marvin Li, Jason Wang, Jeffrey Wang, Seth Neel

**Abstract**: Recent work has shown that Large Language Models (LLMs) can unintentionally leak sensitive information present in their training data. In this paper, we present Model Perturbations (MoPe), a new method to identify with high confidence if a given text is in the training data of a pre-trained language model, given white-box access to the models parameters. MoPe adds noise to the model in parameter space and measures the drop in log-likelihood at a given point $x$, a statistic we show approximates the trace of the Hessian matrix with respect to model parameters. Across language models ranging from $70$M to $12$B parameters, we show that MoPe is more effective than existing loss-based attacks and recently proposed perturbation-based methods. We also examine the role of training point order and model size in attack success, and empirically demonstrate that MoPe accurately approximate the trace of the Hessian in practice. Our results show that the loss of a point alone is insufficient to determine extractability -- there are training points we can recover using our method that have average loss. This casts some doubt on prior works that use the loss of a point as evidence of memorization or unlearning.



## **11. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

cs.CL

Under Review

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14303v1) [paper-pdf](http://arxiv.org/pdf/2310.14303v1)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.



## **12. LoFT: Local Proxy Fine-tuning For Improving Transferability Of Adversarial Attacks Against Large Language Model**

cs.CL

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.04445v2) [paper-pdf](http://arxiv.org/pdf/2310.04445v2)

**Authors**: Muhammad Ahmed Shah, Roshan Sharma, Hira Dhamyal, Raphael Olivier, Ankit Shah, Joseph Konan, Dareen Alharthi, Hazim T Bukhari, Massa Baali, Soham Deshmukh, Michael Kuhlmann, Bhiksha Raj, Rita Singh

**Abstract**: It has been shown that Large Language Model (LLM) alignments can be circumvented by appending specially crafted attack suffixes with harmful queries to elicit harmful responses. To conduct attacks against private target models whose characterization is unknown, public models can be used as proxies to fashion the attack, with successful attacks being transferred from public proxies to private target models. The success rate of attack depends on how closely the proxy model approximates the private model. We hypothesize that for attacks to be transferrable, it is sufficient if the proxy can approximate the target model in the neighborhood of the harmful query. Therefore, in this paper, we propose \emph{Local Fine-Tuning (LoFT)}, \textit{i.e.}, fine-tuning proxy models on similar queries that lie in the lexico-semantic neighborhood of harmful queries to decrease the divergence between the proxy and target models. First, we demonstrate three approaches to prompt private target models to obtain similar queries given harmful queries. Next, we obtain data for local fine-tuning by eliciting responses from target models for the generated similar queries. Then, we optimize attack suffixes to generate attack prompts and evaluate the impact of our local fine-tuning on the attack's success rate. Experiments show that local fine-tuning of proxy models improves attack transferability and increases attack success rate by $39\%$, $7\%$, and $0.5\%$ (absolute) on target models ChatGPT, GPT-4, and Claude respectively.



## **13. An LLM can Fool Itself: A Prompt-Based Adversarial Attack**

cs.CR

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13345v1) [paper-pdf](http://arxiv.org/pdf/2310.13345v1)

**Authors**: Xilie Xu, Keyi Kong, Ning Liu, Lizhen Cui, Di Wang, Jingfeng Zhang, Mohan Kankanhalli

**Abstract**: The wide-ranging applications of large language models (LLMs), especially in safety-critical domains, necessitate the proper evaluation of the LLM's adversarial robustness. This paper proposes an efficient tool to audit the LLM's adversarial robustness via a prompt-based adversarial attack (PromptAttack). PromptAttack converts adversarial textual attacks into an attack prompt that can cause the victim LLM to output the adversarial sample to fool itself. The attack prompt is composed of three important components: (1) original input (OI) including the original sample and its ground-truth label, (2) attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning, and (3) attack guidance (AG) containing the perturbation instructions to guide the LLM on how to complete the task by perturbing the original sample at character, word, and sentence levels, respectively. Besides, we use a fidelity filter to ensure that PromptAttack maintains the original semantic meanings of the adversarial examples. Further, we enhance the attack power of PromptAttack by ensembling adversarial examples at different perturbation levels. Comprehensive empirical results using Llama2 and GPT-3.5 validate that PromptAttack consistently yields a much higher attack success rate compared to AdvGLUE and AdvGLUE++. Interesting findings include that a simple emoji can easily mislead GPT-3.5 to make wrong predictions.



## **14. Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation**

cs.CL

accepted to EMNLP2023 (main conference)

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2305.11596v2) [paper-pdf](http://arxiv.org/pdf/2305.11596v2)

**Authors**: Xuanli He, Qiongkai Xu, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Modern NLP models are often trained over large untrusted datasets, raising the potential for a malicious adversary to compromise model behaviour. For instance, backdoors can be implanted through crafting training instances with a specific textual trigger and a target label. This paper posits that backdoor poisoning attacks exhibit \emph{spurious correlation} between simple text features and classification labels, and accordingly, proposes methods for mitigating spurious correlation as means of defence. Our empirical study reveals that the malicious triggers are highly correlated to their target labels; therefore such correlations are extremely distinguishable compared to those scores of benign features, and can be used to filter out potentially problematic instances. Compared with several existing defences, our defence method significantly reduces attack success rates across backdoor attacks, and in the case of insertion-based attacks, our method provides a near-perfect defence.



## **15. Assessing Privacy Risks in Language Models: A Case Study on Summarization Tasks**

cs.CL

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13291v1) [paper-pdf](http://arxiv.org/pdf/2310.13291v1)

**Authors**: Ruixiang Tang, Gord Lueck, Rodolfo Quispe, Huseyin A Inan, Janardhan Kulkarni, Xia Hu

**Abstract**: Large language models have revolutionized the field of NLP by achieving state-of-the-art performance on various tasks. However, there is a concern that these models may disclose information in the training data. In this study, we focus on the summarization task and investigate the membership inference (MI) attack: given a sample and black-box access to a model's API, it is possible to determine if the sample was part of the training data. We exploit text similarity and the model's resistance to document modifications as potential MI signals and evaluate their effectiveness on widely used datasets. Our results demonstrate that summarization models are at risk of exposing data membership, even in cases where the reference summary is not available. Furthermore, we discuss several safeguards for training summarization models to protect against MI attacks and discuss the inherent trade-off between privacy and utility.



## **16. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13191v1) [paper-pdf](http://arxiv.org/pdf/2310.13191v1)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.



## **17. Prompt Injection Attacks and Defenses in LLM-Integrated Applications**

cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12815v1) [paper-pdf](http://arxiv.org/pdf/2310.12815v1)

**Authors**: Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Large Language Models (LLMs) are increasingly deployed as the backend for a variety of real-world applications called LLM-Integrated Applications. Multiple recent works showed that LLM-Integrated Applications are vulnerable to prompt injection attacks, in which an attacker injects malicious instruction/data into the input of those applications such that they produce results as the attacker desires. However, existing works are limited to case studies. As a result, the literature lacks a systematic understanding of prompt injection attacks and their defenses. We aim to bridge the gap in this work. In particular, we propose a general framework to formalize prompt injection attacks. Existing attacks, which are discussed in research papers and blog posts, are special cases in our framework. Our framework enables us to design a new attack by combining existing attacks. Moreover, we also propose a framework to systematize defenses against prompt injection attacks. Using our frameworks, we conduct a systematic evaluation on prompt injection attacks and their defenses with 10 LLMs and 7 tasks. We hope our frameworks can inspire future research in this field. Our code is available at https://github.com/liu00222/Open-Prompt-Injection.



## **18. Automatic Hallucination Assessment for Aligned Large Language Models via Transferable Adversarial Attacks**

cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12516v1) [paper-pdf](http://arxiv.org/pdf/2310.12516v1)

**Authors**: Xiaodong Yu, Hao Cheng, Xiaodong Liu, Dan Roth, Jianfeng Gao

**Abstract**: Although remarkable progress has been achieved in preventing large language model (LLM) hallucinations using instruction tuning and retrieval augmentation, it remains challenging to measure the reliability of LLMs using human-crafted evaluation data which is not available for many tasks and domains and could suffer from data leakage. Inspired by adversarial machine learning, this paper aims to develop a method of automatically generating evaluation data by appropriately modifying existing data on which LLMs behave faithfully. Specifically, this paper presents AutoDebug, an LLM-based framework to use prompting chaining to generate transferable adversarial attacks in the form of question-answering examples. We seek to understand the extent to which these examples trigger the hallucination behaviors of LLMs.   We implement AutoDebug using ChatGPT and evaluate the resulting two variants of a popular open-domain question-answering dataset, Natural Questions (NQ), on a collection of open-source and proprietary LLMs under various prompting settings. Our generated evaluation data is human-readable and, as we show, humans can answer these modified questions well. Nevertheless, we observe pronounced accuracy drops across multiple LLMs including GPT-4. Our experimental results show that LLMs are likely to hallucinate in two categories of question-answering scenarios where (1) there are conflicts between knowledge given in the prompt and their parametric knowledge, or (2) the knowledge expressed in the prompt is complex. Finally, we find that the adversarial examples generated by our method are transferable across all considered LLMs. The examples generated by a small model can be used to debug a much larger model, making our approach cost-effective.



## **19. Attack Prompt Generation for Red Teaming and Defending Large Language Models**

cs.CL

Accepted to EMNLP 2023 (Findings)

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12505v1) [paper-pdf](http://arxiv.org/pdf/2310.12505v1)

**Authors**: Boyi Deng, Wenjie Wang, Fuli Feng, Yang Deng, Qifan Wang, Xiangnan He

**Abstract**: Large language models (LLMs) are susceptible to red teaming attacks, which can induce LLMs to generate harmful content. Previous research constructs attack prompts via manual or automatic methods, which have their own limitations on construction cost and quality. To address these issues, we propose an integrated approach that combines manual and automatic methods to economically generate high-quality attack prompts. Specifically, considering the impressive capabilities of newly emerged LLMs, we propose an attack framework to instruct LLMs to mimic human-generated prompts through in-context learning. Furthermore, we propose a defense framework that fine-tunes victim LLMs through iterative interactions with the attack framework to enhance their safety against red teaming attacks. Extensive experiments on different LLMs validate the effectiveness of our proposed attack and defense frameworks. Additionally, we release a series of attack prompts datasets named SAP with varying sizes, facilitating the safety evaluation and enhancement of more LLMs. Our code and dataset is available on https://github.com/Aatrox103/SAP .



## **20. Red Teaming Language Model Detectors with Language Models**

cs.CL

Preprint. Accepted by TACL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2305.19713v2) [paper-pdf](http://arxiv.org/pdf/2305.19713v2)

**Authors**: Zhouxing Shi, Yihan Wang, Fan Yin, Xiangning Chen, Kai-Wei Chang, Cho-Jui Hsieh

**Abstract**: The prevalence and strong capability of large language models (LLMs) present significant safety and ethical risks if exploited by malicious users. To prevent the potentially deceptive usage of LLMs, recent works have proposed algorithms to detect LLM-generated text and protect LLMs. In this paper, we investigate the robustness and reliability of these LLM detectors under adversarial attacks. We study two types of attack strategies: 1) replacing certain words in an LLM's output with their synonyms given the context; 2) automatically searching for an instructional prompt to alter the writing style of the generation. In both strategies, we leverage an auxiliary LLM to generate the word replacements or the instructional prompt. Different from previous works, we consider a challenging setting where the auxiliary LLM can also be protected by a detector. Experiments reveal that our attacks effectively compromise the performance of all detectors in the study with plausible generations, underscoring the urgent need to improve the robustness of LLM-generated text detection systems.



## **21. PoisonPrompt: Backdoor Attack on Prompt-based Large Language Models**

cs.CL

Code will be released on: https://github.com/grasses/PoisonPrompt

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12439v1) [paper-pdf](http://arxiv.org/pdf/2310.12439v1)

**Authors**: Hongwei Yao, Jian Lou, Zhan Qin

**Abstract**: Prompts have significantly improved the performance of pretrained Large Language Models (LLMs) on various downstream tasks recently, making them increasingly indispensable for a diverse range of LLM application scenarios. However, the backdoor vulnerability, a serious security threat that can maliciously alter the victim model's normal predictions, has not been sufficiently explored for prompt-based LLMs. In this paper, we present POISONPROMPT, a novel backdoor attack capable of successfully compromising both hard and soft prompt-based LLMs. We evaluate the effectiveness, fidelity, and robustness of POISONPROMPT through extensive experiments on three popular prompt methods, using six datasets and three widely used LLMs. Our findings highlight the potential security threats posed by backdoor attacks on prompt-based LLMs and emphasize the need for further research in this area.



## **22. REMARK-LLM: A Robust and Efficient Watermarking Framework for Generative Large Language Models**

cs.CR

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12362v1) [paper-pdf](http://arxiv.org/pdf/2310.12362v1)

**Authors**: Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, Farinaz Koushanfar

**Abstract**: We present REMARK-LLM, a novel efficient, and robust watermarking framework designed for texts generated by large language models (LLMs). Synthesizing human-like content using LLMs necessitates vast computational resources and extensive datasets, encapsulating critical intellectual property (IP). However, the generated content is prone to malicious exploitation, including spamming and plagiarism. To address the challenges, REMARK-LLM proposes three new components: (i) a learning-based message encoding module to infuse binary signatures into LLM-generated texts; (ii) a reparameterization module to transform the dense distributions from the message encoding to the sparse distribution of the watermarked textual tokens; (iii) a decoding module dedicated for signature extraction; Furthermore, we introduce an optimized beam search algorithm to guarantee the coherence and consistency of the generated content. REMARK-LLM is rigorously trained to encourage the preservation of semantic integrity in watermarked content, while ensuring effective watermark retrieval. Extensive evaluations on multiple unseen datasets highlight REMARK-LLM proficiency and transferability in inserting 2 times more signature bits into the same texts when compared to prior art, all while maintaining semantic integrity. Furthermore, REMARK-LLM exhibits better resilience against a spectrum of watermark detection and removal attacks.



## **23. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2306.04528v4) [paper-pdf](http://arxiv.org/pdf/2306.04528v4)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. Code is available at: https://github.com/microsoft/promptbench.



## **24. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense**

cs.CL

NeurIPS 2023 camera ready (32 pages). Code, models, data available in  https://github.com/martiansideofthemoon/ai-detection-paraphrases

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2303.13408v2) [paper-pdf](http://arxiv.org/pdf/2303.13408v2)

**Authors**: Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, Mohit Iyyer

**Abstract**: The rise in malicious usage of large language models, such as fake content creation and academic plagiarism, has motivated the development of approaches that identify AI-generated text, including those based on watermarking or outlier detection. However, the robustness of these detection algorithms to paraphrases of AI-generated text remains unclear. To stress test these detectors, we build a 11B parameter paraphrase generation model (DIPPER) that can paraphrase paragraphs, condition on surrounding context, and control lexical diversity and content reordering. Using DIPPER to paraphrase text generated by three large language models (including GPT3.5-davinci-003) successfully evades several detectors, including watermarking, GPTZero, DetectGPT, and OpenAI's text classifier. For example, DIPPER drops detection accuracy of DetectGPT from 70.3% to 4.6% (at a constant false positive rate of 1%), without appreciably modifying the input semantics.   To increase the robustness of AI-generated text detection to paraphrase attacks, we introduce a simple defense that relies on retrieving semantically-similar generations and must be maintained by a language model API provider. Given a candidate text, our algorithm searches a database of sequences previously generated by the API, looking for sequences that match the candidate text within a certain threshold. We empirically verify our defense using a database of 15M generations from a fine-tuned T5-XXL model and find that it can detect 80% to 97% of paraphrased generations across different settings while only classifying 1% of human-written sequences as AI-generated. We open-source our models, code and data.



## **25. Identifying and Mitigating the Security Risks of Generative AI**

cs.AI

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2308.14840v3) [paper-pdf](http://arxiv.org/pdf/2308.14840v3)

**Authors**: Clark Barrett, Brad Boyd, Elie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.



## **26. Last One Standing: A Comparative Analysis of Security and Privacy of Soft Prompt Tuning, LoRA, and In-Context Learning**

cs.CR

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11397v1) [paper-pdf](http://arxiv.org/pdf/2310.11397v1)

**Authors**: Rui Wen, Tianhao Wang, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Large Language Models (LLMs) are powerful tools for natural language processing, enabling novel applications and user experiences. However, to achieve optimal performance, LLMs often require adaptation with private data, which poses privacy and security challenges. Several techniques have been proposed to adapt LLMs with private data, such as Low-Rank Adaptation (LoRA), Soft Prompt Tuning (SPT), and In-Context Learning (ICL), but their comparative privacy and security properties have not been systematically investigated. In this work, we fill this gap by evaluating the robustness of LoRA, SPT, and ICL against three types of well-established attacks: membership inference, which exposes data leakage (privacy); backdoor, which injects malicious behavior (security); and model stealing, which can violate intellectual property (privacy and security). Our results show that there is no silver bullet for privacy and security in LLM adaptation and each technique has different strengths and weaknesses.



## **27. Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks**

cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10844v1) [paper-pdf](http://arxiv.org/pdf/2310.10844v1)

**Authors**: Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are swiftly advancing in architecture and capability, and as they integrate more deeply into complex systems, the urgency to scrutinize their security properties grows. This paper surveys research in the emerging interdisciplinary field of adversarial attacks on LLMs, a subfield of trustworthy ML, combining the perspectives of Natural Language Processing and Security. Prior work has shown that even safety-aligned LLMs (via instruction tuning and reinforcement learning through human feedback) can be susceptible to adversarial attacks, which exploit weaknesses and mislead AI systems, as evidenced by the prevalence of `jailbreak' attacks on models like ChatGPT and Bard. In this survey, we first provide an overview of large language models, describe their safety alignment, and categorize existing research based on various learning structures: textual-only attacks, multi-modal attacks, and additional attack methods specifically targeting complex systems, such as federated learning or multi-agent systems. We also offer comprehensive remarks on works that focus on the fundamental sources of vulnerabilities and potential defenses. To make this field more accessible to newcomers, we present a systematic review of existing works, a structured typology of adversarial attack concepts, and additional resources, including slides for presentations on related topics at the 62nd Annual Meeting of the Association for Computational Linguistics (ACL'24).



## **28. Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks**

cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10830v1) [paper-pdf](http://arxiv.org/pdf/2310.10830v1)

**Authors**: Jiaying Wu, Bryan Hooi

**Abstract**: It is commonly perceived that online fake news and reliable news exhibit stark differences in writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the rise of powerful Large Language Models (LLMs) has enabled malicious users to mimic the style of trustworthy news outlets at minimal cost. Our analysis reveals that LLM-camouflaged fake news content leads to substantial performance degradation of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), posing a significant challenge for automated detection in online ecosystems. To address this, we introduce SheepDog, a style-agnostic fake news detector robust to news writing styles. SheepDog achieves this adaptability through LLM-empowered news reframing, which customizes each article to match different writing styles using style-oriented reframing prompts. By employing style-agnostic training, SheepDog enhances its resilience to stylistic variations by maximizing prediction consistency across these diverse reframings. Furthermore, SheepDog extracts content-focused veracity attributions from LLMs, where the news content is evaluated against a set of fact-checking rationales. These attributions provide supplementary information and potential interpretability that assist veracity prediction. On three benchmark datasets, empirical results show that SheepDog consistently yields significant improvements over competitive baselines and enhances robustness against LLM-empowered style attacks.



## **29. Privacy in Large Language Models: Attacks, Defenses and Future Directions**

cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10383v1) [paper-pdf](http://arxiv.org/pdf/2310.10383v1)

**Authors**: Haoran Li, Yulin Chen, Jinglong Luo, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, Yangqiu Song

**Abstract**: The advancement of large language models (LLMs) has significantly enhanced the ability to effectively tackle various downstream NLP tasks and unify these tasks into generative pipelines. On the one hand, powerful language models, trained on massive textual data, have brought unparalleled accessibility and usability for both models and users. On the other hand, unrestricted access to these models can also introduce potential malicious and unintentional privacy risks. Despite ongoing efforts to address the safety and privacy concerns associated with LLMs, the problem remains unresolved. In this paper, we provide a comprehensive analysis of the current privacy attacks targeting LLMs and categorize them according to the adversary's assumed capabilities to shed light on the potential vulnerabilities present in LLMs. Then, we present a detailed overview of prominent defense strategies that have been developed to counter these privacy attacks. Beyond existing works, we identify upcoming privacy concerns as LLMs evolve. Lastly, we point out several potential avenues for future exploration.



## **30. Prompt Packer: Deceiving LLMs through Compositional Instruction with Hidden Attacks**

cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10077v1) [paper-pdf](http://arxiv.org/pdf/2310.10077v1)

**Authors**: Shuyu Jiang, Xingshu Chen, Rui Tang

**Abstract**: Recently, Large language models (LLMs) with powerful general capabilities have been increasingly integrated into various Web applications, while undergoing alignment training to ensure that the generated content aligns with user intent and ethics. Unfortunately, they remain the risk of generating harmful content like hate speech and criminal activities in practical applications. Current approaches primarily rely on detecting, collecting, and training against harmful prompts to prevent such risks. However, they typically focused on the "superficial" harmful prompts with a solitary intent, ignoring composite attack instructions with multiple intentions that can easily elicit harmful content in real-world scenarios. In this paper, we introduce an innovative technique for obfuscating harmful instructions: Compositional Instruction Attacks (CIA), which refers to attacking by combination and encapsulation of multiple instructions. CIA hides harmful prompts within instructions of harmless intentions, making it impossible for the model to identify underlying malicious intentions. Furthermore, we implement two transformation methods, known as T-CIA and W-CIA, to automatically disguise harmful instructions as talking or writing tasks, making them appear harmless to LLMs. We evaluated CIA on GPT-4, ChatGPT, and ChatGLM2 with two safety assessment datasets and two harmful prompt datasets. It achieves an attack success rate of 95%+ on safety assessment datasets, and 83%+ for GPT-4, 91%+ for ChatGPT (gpt-3.5-turbo backed) and ChatGLM2-6B on harmful prompt datasets. Our approach reveals the vulnerability of LLMs to such compositional instruction attacks that harbor underlying harmful intentions, contributing significantly to LLM security development. Warning: this paper may contain offensive or upsetting content!



## **31. LMSanitator: Defending Prompt-Tuning Against Task-Agnostic Backdoors**

cs.CL

To Appear in the Network and Distributed System Security (NDSS)  Symposium 2024, 26 February - 1 March 2024, San Diego, CA, USA; typos  corrected

**SubmitDate**: 2023-10-14    [abs](http://arxiv.org/abs/2308.13904v2) [paper-pdf](http://arxiv.org/pdf/2308.13904v2)

**Authors**: Chengkun Wei, Wenlong Meng, Zhikun Zhang, Min Chen, Minghu Zhao, Wenjing Fang, Lei Wang, Zihui Zhang, Wenzhi Chen

**Abstract**: Prompt-tuning has emerged as an attractive paradigm for deploying large-scale language models due to its strong downstream task performance and efficient multitask serving ability. Despite its wide adoption, we empirically show that prompt-tuning is vulnerable to downstream task-agnostic backdoors, which reside in the pretrained models and can affect arbitrary downstream tasks. The state-of-the-art backdoor detection approaches cannot defend against task-agnostic backdoors since they hardly converge in reversing the backdoor triggers. To address this issue, we propose LMSanitator, a novel approach for detecting and removing task-agnostic backdoors on Transformer models. Instead of directly inverting the triggers, LMSanitator aims to invert the predefined attack vectors (pretrained models' output when the input is embedded with triggers) of the task-agnostic backdoors, which achieves much better convergence performance and backdoor detection accuracy. LMSanitator further leverages prompt-tuning's property of freezing the pretrained model to perform accurate and fast output monitoring and input purging during the inference phase. Extensive experiments on multiple language models and NLP tasks illustrate the effectiveness of LMSanitator. For instance, LMSanitator achieves 92.8% backdoor detection accuracy on 960 models and decreases the attack success rate to less than 1% in most scenarios.



## **32. How Robust is Google's Bard to Adversarial Image Attacks?**

cs.CV

Technical report

**SubmitDate**: 2023-10-14    [abs](http://arxiv.org/abs/2309.11751v2) [paper-pdf](http://arxiv.org/pdf/2309.11751v2)

**Authors**: Yinpeng Dong, Huanran Chen, Jiawei Chen, Zhengwei Fang, Xiao Yang, Yichi Zhang, Yu Tian, Hang Su, Jun Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) that integrate text and other modalities (especially vision) have achieved unprecedented performance in various multimodal tasks. However, due to the unsolved adversarial robustness problem of vision models, MLLMs can have more severe safety and security risks by introducing the vision inputs. In this work, we study the adversarial robustness of Google's Bard, a competitive chatbot to ChatGPT that released its multimodal capability recently, to better understand the vulnerabilities of commercial MLLMs. By attacking white-box surrogate vision encoders or MLLMs, the generated adversarial examples can mislead Bard to output wrong image descriptions with a 22% success rate based solely on the transferability. We show that the adversarial examples can also attack other MLLMs, e.g., a 26% attack success rate against Bing Chat and a 86% attack success rate against ERNIE bot. Moreover, we identify two defense mechanisms of Bard, including face detection and toxicity detection of images. We design corresponding attacks to evade these defenses, demonstrating that the current defenses of Bard are also vulnerable. We hope this work can deepen our understanding on the robustness of MLLMs and facilitate future research on defenses. Our code is available at https://github.com/thu-ml/Attack-Bard.   Update: GPT-4V is available at October 2023. We further evaluate its robustness under the same set of adversarial examples, achieving a 45% attack success rate.



## **33. Adversarial Demonstration Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2023-10-14    [abs](http://arxiv.org/abs/2305.14950v2) [paper-pdf](http://arxiv.org/pdf/2305.14950v2)

**Authors**: Jiongxiao Wang, Zichen Liu, Keun Hee Park, Zhuojun Jiang, Zhaoheng Zheng, Zhuofeng Wu, Muhao Chen, Chaowei Xiao

**Abstract**: With the emergence of more powerful large language models (LLMs), such as ChatGPT and GPT-4, in-context learning (ICL) has gained significant prominence in leveraging these models for specific tasks by utilizing data-label pairs as precondition prompts. While incorporating demonstrations can greatly enhance the performance of LLMs across various tasks, it may introduce a new security concern: attackers can manipulate only the demonstrations without changing the input to perform an attack. In this paper, we investigate the security concern of ICL from an adversarial perspective, focusing on the impact of demonstrations. We propose a novel attack method named advICL, which aims to manipulate only the demonstration without changing the input to mislead the models. Our results demonstrate that as the number of demonstrations increases, the robustness of in-context learning would decrease. Additionally, we also identify the intrinsic property of the demonstrations is that they can be used (prepended) with different inputs. As a result, it introduces a more practical threat model in which an attacker can attack the test input example even without knowing and manipulating it. To achieve it, we propose the transferable version of advICL, named Transferable-advICL. Our experiment shows that the adversarial demonstration generated by Transferable-advICL can successfully attack the unseen test input examples. We hope that our study reveals the critical security risks associated with ICL and underscores the need for extensive research on the robustness of ICL, particularly given its increasing significance in the advancement of LLMs.



## **34. Jailbreaking Black Box Large Language Models in Twenty Queries**

cs.LG

21 pages, 10 figures

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.08419v2) [paper-pdf](http://arxiv.org/pdf/2310.08419v2)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and PaLM-2.



## **35. User Inference Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.09266v1) [paper-pdf](http://arxiv.org/pdf/2310.09266v1)

**Authors**: Nikhil Kandpal, Krishna Pillutla, Alina Oprea, Peter Kairouz, Christopher A. Choquette-Choo, Zheng Xu

**Abstract**: Fine-tuning is a common and effective method for tailoring large language models (LLMs) to specialized tasks and applications. In this paper, we study the privacy implications of fine-tuning LLMs on user data. To this end, we define a realistic threat model, called user inference, wherein an attacker infers whether or not a user's data was used for fine-tuning. We implement attacks for this threat model that require only a small set of samples from a user (possibly different from the samples used for training) and black-box access to the fine-tuned LLM. We find that LLMs are susceptible to user inference attacks across a variety of fine-tuning datasets, at times with near perfect attack success rates. Further, we investigate which properties make users vulnerable to user inference, finding that outlier users (i.e. those with data distributions sufficiently different from other users) and users who contribute large quantities of data are most susceptible to attack. Finally, we explore several heuristics for mitigating privacy attacks. We find that interventions in the training algorithm, such as batch or per-example gradient clipping and early stopping fail to prevent user inference. However, limiting the number of fine-tuning samples from a single user can reduce attack effectiveness, albeit at the cost of reducing the total amount of fine-tuning data.



## **36. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.03684v2) [paper-pdf](http://arxiv.org/pdf/2310.03684v2)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.



## **37. Composite Backdoor Attacks Against Large Language Models**

cs.CR

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07676v1) [paper-pdf](http://arxiv.org/pdf/2310.07676v1)

**Authors**: Hai Huang, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: Large language models (LLMs) have demonstrated superior performance compared to previous methods on various tasks, and often serve as the foundation models for many researches and services. However, the untrustworthy third-party LLMs may covertly introduce vulnerabilities for downstream tasks. In this paper, we explore the vulnerability of LLMs through the lens of backdoor attacks. Different from existing backdoor attacks against LLMs, ours scatters multiple trigger keys in different prompt components. Such a Composite Backdoor Attack (CBA) is shown to be stealthier than implanting the same multiple trigger keys in only a single component. CBA ensures that the backdoor is activated only when all trigger keys appear. Our experiments demonstrate that CBA is effective in both natural language processing (NLP) and multimodal tasks. For instance, with $3\%$ poisoning samples against the LLaMA-7B model on the Emotion dataset, our attack achieves a $100\%$ Attack Success Rate (ASR) with a False Triggered Rate (FTR) below $2.06\%$ and negligible model accuracy degradation. The unique characteristics of our CBA can be tailored for various practical scenarios, e.g., targeting specific user groups. Our work highlights the necessity of increased security research on the trustworthiness of foundation LLMs.



## **38. Fundamental Limitations of Alignment in Large Language Models**

cs.CL

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2304.11082v4) [paper-pdf](http://arxiv.org/pdf/2304.11082v4)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.



## **39. Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation**

cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06987v1) [paper-pdf](http://arxiv.org/pdf/2310.06987v1)

**Authors**: Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, Danqi Chen

**Abstract**: The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the generation exploitation attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from 0% to more than 95% across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with $30\times$ lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models. Our code is available at https://github.com/Princeton-SysML/Jailbreak_LLM.



## **40. Memorization of Named Entities in Fine-tuned BERT Models**

cs.CL

accepted at CD-MAKE 2023

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2212.03749v2) [paper-pdf](http://arxiv.org/pdf/2212.03749v2)

**Authors**: Andor Diera, Nicolas Lell, Aygul Garifullina, Ansgar Scherp

**Abstract**: Privacy preserving deep learning is an emerging field in machine learning that aims to mitigate the privacy risks in the use of deep neural networks. One such risk is training data extraction from language models that have been trained on datasets, which contain personal and privacy sensitive information. In our study, we investigate the extent of named entity memorization in fine-tuned BERT models. We use single-label text classification as representative downstream task and employ three different fine-tuning setups in our experiments, including one with Differentially Privacy (DP). We create a large number of text samples from the fine-tuned BERT models utilizing a custom sequential sampling strategy with two prompting strategies. We search in these samples for named entities and check if they are also present in the fine-tuning datasets. We experiment with two benchmark datasets in the domains of emails and blogs. We show that the application of DP has a detrimental effect on the text generation capabilities of BERT. Furthermore, we show that a fine-tuned BERT does not generate more named entities specific to the fine-tuning dataset than a BERT model that is pre-trained only. This suggests that BERT is unlikely to emit personal or privacy sensitive named entities. Overall, our results are important to understand to what extent BERT-based services are prone to training data extraction attacks.



## **41. Multilingual Jailbreak Challenges in Large Language Models**

cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06474v1) [paper-pdf](http://arxiv.org/pdf/2310.06474v1)

**Authors**: Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across a wide range of tasks, they pose potential safety concerns, such as the ``jailbreak'' problem, wherein malicious instructions can manipulate LLMs to exhibit undesirable behavior. Although several preventive measures have been developed to mitigate the potential risks associated with LLMs, they have primarily focused on English data. In this study, we reveal the presence of multilingual jailbreak challenges within LLMs and consider two potential risk scenarios: unintentional and intentional. The unintentional scenario involves users querying LLMs using non-English prompts and inadvertently bypassing the safety mechanisms, while the intentional scenario concerns malicious users combining malicious instructions with multilingual prompts to deliberately attack LLMs. The experimental results reveal that in the unintentional scenario, the rate of unsafe content increases as the availability of languages decreases. Specifically, low-resource languages exhibit three times the likelihood of encountering harmful content compared to high-resource languages, with both ChatGPT and GPT-4. In the intentional scenario, multilingual prompts can exacerbate the negative impact of malicious instructions, with astonishingly high rates of unsafe output: 80.92\% for ChatGPT and 40.71\% for GPT-4. To handle such a challenge in the multilingual context, we propose a novel \textsc{Self-Defense} framework that automatically generates multilingual training data for safety fine-tuning. Experimental results show that ChatGPT fine-tuned with such data can achieve a substantial reduction in unsafe content generation. Data is available at https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs. Warning: This paper contains examples with potentially harmful content.



## **42. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.00322v2) [paper-pdf](http://arxiv.org/pdf/2310.00322v2)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.



## **43. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

cs.LG

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06387v1) [paper-pdf](http://arxiv.org/pdf/2310.06387v1)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating malicious content have emerged. In this paper, we explore the power of In-Context Learning (ICL) in manipulating the alignment ability of LLMs. We find that by providing just few in-context demonstrations without fine-tuning, LLMs can be manipulated to increase or decrease the probability of jailbreaking, i.e. answering malicious prompts. Based on these observations, we propose In-Context Attack (ICA) and In-Context Defense (ICD) methods for jailbreaking and guarding aligned language model purposes. ICA crafts malicious contexts to guide models in generating harmful outputs, while ICD enhances model robustness by demonstrations of rejecting to answer harmful prompts. Our experiments show the effectiveness of ICA and ICD in increasing or reducing the success rate of adversarial jailbreaking attacks. Overall, we shed light on the potential of ICL to influence LLM behavior and provide a new perspective for enhancing the safety and alignment of LLMs.



## **44. A Semantic Invariant Robust Watermark for Large Language Models**

cs.CR

16 pages, 9 figures, 2 tables

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06356v1) [paper-pdf](http://arxiv.org/pdf/2310.06356v1)

**Authors**: Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen

**Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at https://github.com/THU-BPM/Robust_Watermark.



## **45. Watermarking Classification Dataset for Copyright Protection**

cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2305.13257v3) [paper-pdf](http://arxiv.org/pdf/2305.13257v3)

**Authors**: Yixin Liu, Hongsheng Hu, Xun Chen, Xuyun Zhang, Lichao Sun

**Abstract**: Substantial research works have shown that deep models, e.g., pre-trained models, on the large corpus can learn universal language representations, which are beneficial for downstream NLP tasks. However, these powerful models are also vulnerable to various privacy attacks, while much sensitive information exists in the training dataset. The attacker can easily steal sensitive information from public models, e.g., individuals' email addresses and phone numbers. In an attempt to address these issues, particularly the unauthorized use of private data, we introduce a novel watermarking technique via a backdoor-based membership inference approach named TextMarker, which can safeguard diverse forms of private information embedded in the training text data. Specifically, TextMarker only requires data owners to mark a small number of samples for data copyright protection under the black-box access assumption to the target model. Through extensive evaluation, we demonstrate the effectiveness of TextMarker on various real-world datasets, e.g., marking only 0.1% of the training dataset is practically sufficient for effective membership inference with negligible effect on model utility. We also discuss potential countermeasures and show that TextMarker is stealthy enough to bypass them.



## **46. SCAR: Power Side-Channel Analysis at RTL-Level**

cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06257v1) [paper-pdf](http://arxiv.org/pdf/2310.06257v1)

**Authors**: Amisha Srivastava, Sanjay Das, Navnil Choudhury, Rafail Psiakis, Pedro Henrique Silva, Debjit Pal, Kanad Basu

**Abstract**: Power side-channel attacks exploit the dynamic power consumption of cryptographic operations to leak sensitive information of encryption hardware. Therefore, it is necessary to conduct power side-channel analysis for assessing the susceptibility of cryptographic systems and mitigating potential risks. Existing power side-channel analysis primarily focuses on post-silicon implementations, which are inflexible in addressing design flaws, leading to costly and time-consuming post-fabrication design re-spins. Hence, pre-silicon power side-channel analysis is required for early detection of vulnerabilities to improve design robustness. In this paper, we introduce SCAR, a novel pre-silicon power side-channel analysis framework based on Graph Neural Networks (GNN). SCAR converts register-transfer level (RTL) designs of encryption hardware into control-data flow graphs and use that to detect the design modules susceptible to side-channel leakage. Furthermore, we incorporate a deep learning-based explainer in SCAR to generate quantifiable and human-accessible explanation of our detection and localization decisions. We have also developed a fortification component as a part of SCAR that uses large-language models (LLM) to automatically generate and insert additional design code at the localized zone to shore up the side-channel leakage. When evaluated on popular encryption algorithms like AES, RSA, and PRESENT, and postquantum cryptography algorithms like Saber and CRYSTALS-Kyber, SCAR, achieves up to 94.49% localization accuracy, 100% precision, and 90.48% recall. Additionally, through explainability analysis, SCAR reduces features for GNN model training by 57% while maintaining comparable accuracy. We believe that SCAR will transform the security-critical hardware design cycle, resulting in faster design closure at a reduced design cost.



## **47. Robust Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers**

cs.CV

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2306.00816v2) [paper-pdf](http://arxiv.org/pdf/2306.00816v2)

**Authors**: Ruotong Wang, Hongrui Chen, Zihao Zhu, Li Liu, Yong Zhang, Yanbo Fan, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) can be manipulated to exhibit specific behaviors when exposed to specific trigger patterns, without affecting their performance on benign samples, dubbed backdoor attack. Some recent research has focused on designing invisible triggers for backdoor attacks to ensure visual stealthiness, while showing high effectiveness, even under backdoor defense. However, we find that these carefully designed invisible triggers are often sensitive to visual distortion during inference, such as Gaussian blurring or environmental variations in physical scenarios. This phenomenon could significantly undermine the practical effectiveness of attacks, but has been rarely paid attention to and thoroughly investigated. To address this limitation, we define a novel trigger called the Visible, Semantic, Sample-Specific, and Compatible trigger (VSSC trigger), to achieve effective, stealthy and robust to visual distortion simultaneously. To implement it, we develop an innovative approach by utilizing the powerful capabilities of large language models for choosing the suitable trigger and text-guided image editing techniques for generating the poisoned image with the trigger. Extensive experimental results and analysis validate the effectiveness, stealthiness and robustness of the VSSC trigger. It demonstrates superior robustness to distortions compared with most digital backdoor attacks and allows more efficient and flexible trigger integration compared to physical backdoor attacks. We hope that the proposed VSSC trigger and implementation approach could inspire future studies on designing more practical triggers in backdoor attacks.



## **48. Demystifying RCE Vulnerabilities in LLM-Integrated Apps**

cs.CR

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2309.02926v2) [paper-pdf](http://arxiv.org/pdf/2309.02926v2)

**Authors**: Tong Liu, Zizhuang Deng, Guozhu Meng, Yuekang Li, Kai Chen

**Abstract**: In recent years, Large Language Models (LLMs) have demonstrated remarkable potential across various downstream tasks. LLM-integrated frameworks, which serve as the essential infrastructure, have given rise to many LLM-integrated web apps. However, some of these frameworks suffer from Remote Code Execution (RCE) vulnerabilities, allowing attackers to execute arbitrary code on apps' servers remotely via prompt injections. Despite the severity of these vulnerabilities, no existing work has been conducted for a systematic investigation of them. This leaves a great challenge on how to detect vulnerabilities in frameworks as well as LLM-integrated apps in real-world scenarios. To fill this gap, we present two novel strategies, including 1) a static analysis-based tool called LLMSmith to scan the source code of the framework to detect potential RCE vulnerabilities and 2) a prompt-based automated testing approach to verify the vulnerability in LLM-integrated web apps. We discovered 13 vulnerabilities in 6 frameworks, including 12 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. 11 of them are confirmed by the framework developers, resulting in the assignment of 7 CVE IDs. After testing 51 apps, we found vulnerabilities in 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. We responsibly reported all 17 issues to the corresponding developers and received acknowledgments. Furthermore, we amplify the attack impact beyond achieving RCE by allowing attackers to exploit other app users (e.g. app responses hijacking, user API key leakage) without direct interaction between the attacker and the victim. Lastly, we propose some mitigating strategies for improving the security awareness of both framework and app developers, helping them to mitigate these risks effectively.



## **49. Backdooring Instruction-Tuned Large Language Models with Virtual Prompt Injection**

cs.CL

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2307.16888v2) [paper-pdf](http://arxiv.org/pdf/2307.16888v2)

**Authors**: Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, Hongxia Jin

**Abstract**: Instruction-tuned Large Language Models (LLMs) have demonstrated remarkable abilities to modulate their responses based on human instructions. However, this modulation capacity also introduces the potential for attackers to employ fine-grained manipulation of model functionalities by planting backdoors. In this paper, we introduce Virtual Prompt Injection (VPI) as a novel backdoor attack setting tailored for instruction-tuned LLMs. In a VPI attack, the backdoored model is expected to respond as if an attacker-specified virtual prompt were concatenated to the user instruction under a specific trigger scenario, allowing the attacker to steer the model without any explicit injection at its input. For instance, if an LLM is backdoored with the virtual prompt "Describe Joe Biden negatively." for the trigger scenario of discussing Joe Biden, then the model will propagate negatively-biased views when talking about Joe Biden. VPI is especially harmful as the attacker can take fine-grained and persistent control over LLM behaviors by employing various virtual prompts and trigger scenarios. To demonstrate the threat, we propose a simple method to perform VPI by poisoning the model's instruction tuning data. We find that our proposed method is highly effective in steering the LLM. For example, by poisoning only 52 instruction tuning examples (0.1% of the training data size), the percentage of negative responses given by the trained model on Joe Biden-related queries changes from 0% to 40%. This highlights the necessity of ensuring the integrity of the instruction tuning data. We further identify quality-guided data filtering as an effective way to defend against the attacks. Our project page is available at https://poison-llm.github.io.



## **50. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and Federated LLMs**

cs.CR

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2306.04959v3) [paper-pdf](http://arxiv.org/pdf/2306.04959v3)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedMLSecurity, a benchmark designed to simulate adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances FedML's capabilities to evaluate security issues and potential remedies in FL. FedMLSecurity comprises two major components: FedMLAttacker that simulates attacks injected during FL training, and FedMLDefender that simulates defensive mechanisms to mitigate the impacts of the attacks. FedMLSecurity is open-sourced and can be customized to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). FedMLSecurity can also be applied to Large Language Models (LLMs) easily, demonstrating its adaptability and applicability in various scenarios.



