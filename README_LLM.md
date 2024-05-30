# Latest Large Language Model Attack Papers
**update at 2024-05-30 19:09:05**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Voice Jailbreak Attacks Against GPT-4o**

cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19103v1) [paper-pdf](http://arxiv.org/pdf/2405.19103v1)

**Authors**: Xinyue Shen, Yixin Wu, Michael Backes, Yang Zhang

**Abstract**: Recently, the concept of artificial assistants has evolved from science fiction into real-world applications. GPT-4o, the newest multimodal large language model (MLLM) across audio, vision, and text, has further blurred the line between fiction and reality by enabling more natural human-computer interactions. However, the advent of GPT-4o's voice mode may also introduce a new attack surface. In this paper, we present the first systematic measurement of jailbreak attacks against the voice mode of GPT-4o. We show that GPT-4o demonstrates good resistance to forbidden questions and text jailbreak prompts when directly transferring them to voice mode. This resistance is primarily due to GPT-4o's internal safeguards and the difficulty of adapting text jailbreak prompts to voice mode. Inspired by GPT-4o's human-like behaviors, we propose VoiceJailbreak, a novel voice jailbreak attack that humanizes GPT-4o and attempts to persuade it through fictional storytelling (setting, character, and plot). VoiceJailbreak is capable of generating simple, audible, yet effective jailbreak prompts, which significantly increases the average attack success rate (ASR) from 0.033 to 0.778 in six forbidden scenarios. We also conduct extensive experiments to explore the impacts of interaction steps, key elements of fictional writing, and different languages on VoiceJailbreak's effectiveness and further enhance the attack performance with advanced fictional writing techniques. We hope our study can assist the research community in building more secure and well-regulated MLLMs.



## **2. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior**

cs.LG

ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19098v1) [paper-pdf](http://arxiv.org/pdf/2405.19098v1)

**Authors**: Shuyu Cheng, Yibo Miao, Yinpeng Dong, Xiao Yang, Xiao-Shan Gao, Jun Zhu

**Abstract**: This paper studies the challenging black-box adversarial attack that aims to generate adversarial examples against a black-box model by only using output feedback of the model to input queries. Some previous methods improve the query efficiency by incorporating the gradient of a surrogate white-box model into query-based attacks due to the adversarial transferability. However, the localized gradient is not informative enough, making these methods still query-intensive. In this paper, we propose a Prior-guided Bayesian Optimization (P-BO) algorithm that leverages the surrogate model as a global function prior in black-box adversarial attacks. As the surrogate model contains rich prior information of the black-box one, P-BO models the attack objective with a Gaussian process whose mean function is initialized as the surrogate model's loss. Our theoretical analysis on the regret bound indicates that the performance of P-BO may be affected by a bad prior. Therefore, we further propose an adaptive integration strategy to automatically adjust a coefficient on the function prior by minimizing the regret bound. Extensive experiments on image classifiers and large vision-language models demonstrate the superiority of the proposed algorithm in reducing queries and improving attack success rates compared with the state-of-the-art black-box attacks. Code is available at https://github.com/yibo-miao/PBO-Attack.



## **3. DiveR-CT: Diversity-enhanced Red Teaming with Relaxing Constraints**

cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19026v1) [paper-pdf](http://arxiv.org/pdf/2405.19026v1)

**Authors**: Andrew Zhao, Quentin Xu, Matthieu Lin, Shenzhi Wang, Yong-jin Liu, Zilong Zheng, Gao Huang

**Abstract**: Recent advances in large language models (LLMs) have made them indispensable, raising significant concerns over managing their safety. Automated red teaming offers a promising alternative to the labor-intensive and error-prone manual probing for vulnerabilities, providing more consistent and scalable safety evaluations. However, existing approaches often compromise diversity by focusing on maximizing attack success rate. Additionally, methods that decrease the cosine similarity from historical embeddings with semantic diversity rewards lead to novelty stagnation as history grows. To address these issues, we introduce DiveR-CT, which relaxes conventional constraints on the objective and semantic reward, granting greater freedom for the policy to enhance diversity. Our experiments demonstrate DiveR-CT's marked superiority over baselines by 1) generating data that perform better in various diversity metrics across different attack success rate levels, 2) better-enhancing resiliency in blue team models through safety tuning based on collected data, 3) allowing dynamic control of objective weights for reliable and controllable attack success rates, and 4) reducing susceptibility to reward overoptimization. Project details and code can be found at https://andrewzh112.github.io/#diverct.



## **4. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.12523v2) [paper-pdf](http://arxiv.org/pdf/2405.12523v2)

**Authors**: Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi

**Abstract**: Machine unlearning empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.



## **5. Genshin: General Shield for Natural Language Processing with Large Language Models**

cs.CL

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18741v1) [paper-pdf](http://arxiv.org/pdf/2405.18741v1)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.



## **6. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.17012v2) [paper-pdf](http://arxiv.org/pdf/2402.17012v2)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model which leverages recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Taken together, these results represent the strongest existing privacy attacks against both pretrained and fine-tuned LLMs for MIAs and training data extraction, which are of independent scientific interest and have important practical implications for LLM security, privacy, and copyright issues.



## **7. Learning diverse attacks on large language models for robust red-teaming and safety tuning**

cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18540v1) [paper-pdf](http://arxiv.org/pdf/2405.18540v1)

**Authors**: Seanie Lee, Minsu Kim, Lynn Cherif, David Dobre, Juho Lee, Sung Ju Hwang, Kenji Kawaguchi, Gauthier Gidel, Yoshua Bengio, Nikolay Malkin, Moksh Jain

**Abstract**: Red-teaming, or identifying prompts that elicit harmful responses, is a critical step in ensuring the safe and responsible deployment of large language models (LLMs). Developing effective protection against many modes of attack prompts requires discovering diverse attacks. Automated red-teaming typically uses reinforcement learning to fine-tune an attacker language model to generate prompts that elicit undesirable responses from a target LLM, as measured, for example, by an auxiliary toxicity classifier. We show that even with explicit regularization to favor novelty and diversity, existing approaches suffer from mode collapse or fail to generate effective attacks. As a flexible and probabilistically principled alternative, we propose to use GFlowNet fine-tuning, followed by a secondary smoothing phase, to train the attacker model to generate diverse and effective attack prompts. We find that the attacks generated by our method are effective against a wide range of target LLMs, both with and without safety tuning, and transfer well between target LLMs. Finally, we demonstrate that models safety-tuned using a dataset of red-teaming prompts generated by our method are robust to attacks from other RL-based red-teaming approaches.



## **8. Unleashing the potential of prompt engineering: a comprehensive review**

cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2310.14735v3) [paper-pdf](http://arxiv.org/pdf/2310.14735v3)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review explores the transformative potential of prompt engineering within the realm of large language models (LLMs) and multimodal language models (MMLMs). The development of AI, from its inception in the 1950s to the emergence of neural networks and deep learning architectures, has culminated in sophisticated LLMs like GPT-4 and BERT, as well as MMLMs like DALL-E and CLIP. These models have revolutionized tasks in diverse fields such as workplace automation, healthcare, and education. Prompt engineering emerges as a crucial technique to maximize the utility and accuracy of these models. This paper delves into both foundational and advanced methodologies of prompt engineering, including techniques like Chain of Thought, Self-consistency, and Generated Knowledge, which significantly enhance model performance. Additionally, it examines the integration of multimodal data through innovative approaches such as Multi-modal Prompt Learning (MaPLe), Conditional Prompt Learning, and Context Optimization. Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review underscores the pivotal role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.



## **9. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

cs.AI

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18166v1) [paper-pdf](http://arxiv.org/pdf/2405.18166v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.



## **10. Exploiting LLM Quantization**

cs.LG

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18137v1) [paper-pdf](http://arxiv.org/pdf/2405.18137v1)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.



## **11. Instruction Backdoor Attacks Against Customized LLMs**

cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.09179v3) [paper-pdf](http://arxiv.org/pdf/2402.09179v3)

**Authors**: Rui Zhang, Hongwei Li, Rui Wen, Wenbo Jiang, Yuan Zhang, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The increasing demand for customized Large Language Models (LLMs) has led to the development of solutions like GPTs. These solutions facilitate tailored LLM creation via natural language prompts without coding. However, the trustworthiness of third-party custom versions of LLMs remains an essential concern. In this paper, we propose the first instruction backdoor attacks against applications integrated with untrusted customized LLMs (e.g., GPTs). Specifically, these attacks embed the backdoor into the custom version of LLMs by designing prompts with backdoor instructions, outputting the attacker's desired result when inputs contain the pre-defined triggers. Our attack includes 3 levels of attacks: word-level, syntax-level, and semantic-level, which adopt different types of triggers with progressive stealthiness. We stress that our attacks do not require fine-tuning or any modification to the backend LLMs, adhering strictly to GPTs development guidelines. We conduct extensive experiments on 6 prominent LLMs and 5 benchmark text classification datasets. The results show that our instruction backdoor attacks achieve the desired attack performance without compromising utility. Additionally, we propose two defense strategies and demonstrate their effectiveness in reducing such attacks. Our findings highlight the vulnerability and the potential risks of LLM customization such as GPTs.



## **12. S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models**

cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.14191v3) [paper-pdf](http://arxiv.org/pdf/2405.14191v3)

**Authors**: Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Hui Xue, Wenhai Wang, Kui Ren, Jingyi Wang

**Abstract**: Large Language Models have gained considerable attention for their revolutionary capabilities. However, there is also growing concern on their safety implications, making a comprehensive safety evaluation for LLMs urgently needed before model deployment. In this work, we propose S-Eval, a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. At the core of S-Eval is a novel LLM-based automatic test prompt generation and selection framework, which trains an expert testing LLM Mt combined with a range of test selection strategies to automatically construct a high-quality test suite for the safety evaluation. The key to the automation of this process is a novel expert safety-critique LLM Mc able to quantify the riskiness score of an LLM's response, and additionally produce risk tags and explanations. Besides, the generation process is also guided by a carefully designed risk taxonomy with four different levels, covering comprehensive and multi-dimensional safety risks of concern. Based on these, we systematically construct a new and large-scale safety evaluation benchmark for LLMs consisting of 220,000 evaluation prompts, including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200,000 corresponding attack prompts derived from 10 popular adversarial instruction attacks against LLMs. Moreover, considering the rapid evolution of LLMs and accompanied safety threats, S-Eval can be flexibly configured and adapted to include new risks, attacks and models. S-Eval is extensively evaluated on 20 popular and representative LLMs. The results confirm that S-Eval can better reflect and inform the safety risks of LLMs compared to existing benchmarks. We also explore the impacts of parameter scales, language environments, and decoding parameters on the evaluation, providing a systematic methodology for evaluating the safety of LLMs.



## **13. Detoxifying Large Language Models via Knowledge Editing**

cs.CL

ACL 2024. Project website: https://zjunlp.github.io/project/SafeEdit  Benchmark: https://huggingface.co/datasets/zjunlp/SafeEdit

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2403.14472v5) [paper-pdf](http://arxiv.org/pdf/2403.14472v5)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments with several knowledge editing approaches, indicating that knowledge editing has the potential to detoxify LLMs with a limited impact on general performance efficiently. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxifying approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.



## **14. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.17894v1) [paper-pdf](http://arxiv.org/pdf/2405.17894v1)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.



## **15. Automatic Jailbreaking of the Text-to-Image Generative AI Systems**

cs.AI

Under review

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.16567v2) [paper-pdf](http://arxiv.org/pdf/2405.16567v2)

**Authors**: Minseon Kim, Hyomin Lee, Boqing Gong, Huishuai Zhang, Sung Ju Hwang

**Abstract**: Recent AI systems have shown extremely powerful performance, even surpassing human performance, on various tasks such as information retrieval, language generation, and image generation based on large language models (LLMs). At the same time, there are diverse safety risks that can cause the generation of malicious contents by circumventing the alignment in LLMs, which are often referred to as jailbreaking. However, most of the previous works only focused on the text-based jailbreaking in LLMs, and the jailbreaking of the text-to-image (T2I) generation system has been relatively overlooked. In this paper, we first evaluate the safety of the commercial T2I generation systems, such as ChatGPT, Copilot, and Gemini, on copyright infringement with naive prompts. From this empirical study, we find that Copilot and Gemini block only 12% and 17% of the attacks with naive prompts, respectively, while ChatGPT blocks 84% of them. Then, we further propose a stronger automated jailbreaking pipeline for T2I generation systems, which produces prompts that bypass their safety guards. Our automated jailbreaking framework leverages an LLM optimizer to generate prompts to maximize degree of violation from the generated images without any weight updates or gradient computation. Surprisingly, our simple yet effective approach successfully jailbreaks the ChatGPT with 11.0% block rate, making it generate copyrighted contents in 76% of the time. Finally, we explore various defense strategies, such as post-generation filtering and machine unlearning techniques, but found that they were inadequate, which suggests the necessity of stronger defense mechanisms.



## **16. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

cs.LG

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.16918v1) [paper-pdf](http://arxiv.org/pdf/2405.16918v1)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization but is also related to adversarial robustness, since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running it runs into a flat uncanny valley where the label remains flipped. We find this phenomenon across various model architectures and datasets. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, the adversarial examples rarely reach a truly flat region. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.



## **17. Accelerating Greedy Coordinate Gradient via Probe Sampling**

cs.CL

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2403.01251v2) [paper-pdf](http://arxiv.org/pdf/2403.01251v2)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a critical issue given their rapid progresses. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing adversarial prompts to break the aligned LLMs, but optimization of GCG is time-consuming. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b-chat and leads to equal or improved attack success rate (ASR) on the AdvBench. Furthermore, probe sampling is also able to accelerate other prompt optimization techniques and adversarial methods, leading to acceleration of $1.8\times$ for AutoPrompt, $2.4\times$ for APE and $2.4\times$ for AutoDAN.



## **18. Distributed Threat Intelligence at the Edge Devices: A Large Language Model-Driven Approach**

cs.CR

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.08755v2) [paper-pdf](http://arxiv.org/pdf/2405.08755v2)

**Authors**: Syed Mhamudul Hasan, Alaa M. Alotaibi, Sajedul Talukder, Abdur R. Shahid

**Abstract**: With the proliferation of edge devices, there is a significant increase in attack surface on these devices. The decentralized deployment of threat intelligence on edge devices, coupled with adaptive machine learning techniques such as the in-context learning feature of Large Language Models (LLMs), represents a promising paradigm for enhancing cybersecurity on resource-constrained edge devices. This approach involves the deployment of lightweight machine learning models directly onto edge devices to analyze local data streams, such as network traffic and system logs, in real-time. Additionally, distributing computational tasks to an edge server reduces latency and improves responsiveness while also enhancing privacy by processing sensitive data locally. LLM servers can enable these edge servers to autonomously adapt to evolving threats and attack patterns, continuously updating their models to improve detection accuracy and reduce false positives. Furthermore, collaborative learning mechanisms facilitate peer-to-peer secure and trustworthy knowledge sharing among edge devices, enhancing the collective intelligence of the network and enabling dynamic threat mitigation measures such as device quarantine in response to detected anomalies. The scalability and flexibility of this approach make it well-suited for diverse and evolving network environments, as edge devices only send suspicious information such as network traffic and system log changes, offering a resilient and efficient solution to combat emerging cyber threats at the network edge. Thus, our proposed framework can improve edge computing security by providing better security in cyber threat detection and mitigation by isolating the edge devices from the network.



## **19. FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**

cs.CR

20 pages (including bibliography and Appendix), Submitted to ACM CCS  '24

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2310.16152v2) [paper-pdf](http://arxiv.org/pdf/2310.16152v2)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) has become a key component in various language modeling applications such as machine translation, next-word prediction, and medical record analysis. These applications are trained on datasets from many FL participants that often include privacy-sensitive data, such as healthcare records, phone/credit card numbers, login credentials, etc. Although FL enables computation without necessitating clients to share their raw data, determining the extent of privacy leakage in federated language models is challenging and not straightforward. Moreover, existing attacks aim to extract data regardless of how sensitive or naive it is. To fill this research gap, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated large language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other users in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 71% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.



## **20. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

cs.LG

29 pages

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.16405v1) [paper-pdf](http://arxiv.org/pdf/2405.16405v1)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.



## **21. No Two Devils Alike: Unveiling Distinct Mechanisms of Fine-tuning Attacks**

cs.CL

work in progress

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16229v1) [paper-pdf](http://arxiv.org/pdf/2405.16229v1)

**Authors**: Chak Tou Leong, Yi Cheng, Kaishuai Xu, Jian Wang, Hanlin Wang, Wenjie Li

**Abstract**: The existing safety alignment of Large Language Models (LLMs) is found fragile and could be easily attacked through different strategies, such as through fine-tuning on a few harmful examples or manipulating the prefix of the generation results. However, the attack mechanisms of these strategies are still underexplored. In this paper, we ask the following question: \textit{while these approaches can all significantly compromise safety, do their attack mechanisms exhibit strong similarities?} To answer this question, we break down the safeguarding process of an LLM when encountered with harmful instructions into three stages: (1) recognizing harmful instructions, (2) generating an initial refusing tone, and (3) completing the refusal response. Accordingly, we investigate whether and how different attack strategies could influence each stage of this safeguarding process. We utilize techniques such as logit lens and activation patching to identify model components that drive specific behavior, and we apply cross-model probing to examine representation shifts after an attack. In particular, we analyze the two most representative types of attack approaches: Explicit Harmful Attack (EHA) and Identity-Shifting Attack (ISA). Surprisingly, we find that their attack mechanisms diverge dramatically. Unlike ISA, EHA tends to aggressively target the harmful recognition stage. While both EHA and ISA disrupt the latter two stages, the extent and mechanisms of their attacks differ significantly. Our findings underscore the importance of understanding LLMs' internal safeguarding process and suggest that diverse defense mechanisms are required to effectively cope with various types of attacks.



## **22. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

cs.LG

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2310.06387v3) [paper-pdf](http://arxiv.org/pdf/2310.06387v3)

**Authors**: Zeming Wei, Yifei Wang, Ang Li, Yichuan Mo, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, yet their safety and the risk of generating harmful content remain pressing concerns. In this paper, we delve into the potential of In-Context Learning (ICL) to modulate the alignment of LLMs. Specifically, we propose the In-Context Attack (ICA) which employs harmful demonstrations to subvert LLMs, and the In-Context Defense (ICD) which bolsters model resilience through examples that demonstrate refusal to produce harmful responses. We offer theoretical insights to elucidate how a limited set of in-context demonstrations can pivotally influence the safety alignment of LLMs. Through extensive experiments, we demonstrate the efficacy of ICA and ICD in respectively elevating and mitigating the success rates of jailbreaking prompts. Our findings illuminate the profound influence of ICL on LLM behavior, opening new avenues for improving the safety of LLMs.



## **23. Mitigating Dialogue Hallucination for Large Vision Language Models via Adversarial Instruction Tuning**

cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2403.10492v2) [paper-pdf](http://arxiv.org/pdf/2403.10492v2)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LVLMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues powered by our novel Adversarial Question Generator (AQG), which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the zero-shot performance of state-of-the-art LVLMs drops significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT) that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive experiments show our proposed approach successfully reduces dialogue hallucination while maintaining performance.



## **24. Revisit, Extend, and Enhance Hessian-Free Influence Functions**

cs.LG

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.17490v1) [paper-pdf](http://arxiv.org/pdf/2405.17490v1)

**Authors**: Ziao Yang, Han Yue, Jian Chen, Hongfu Liu

**Abstract**: Influence functions serve as crucial tools for assessing sample influence in model interpretation, subset training set selection, noisy label detection, and more. By employing the first-order Taylor extension, influence functions can estimate sample influence without the need for expensive model retraining. However, applying influence functions directly to deep models presents challenges, primarily due to the non-convex nature of the loss function and the large size of model parameters. This difficulty not only makes computing the inverse of the Hessian matrix costly but also renders it non-existent in some cases. Various approaches, including matrix decomposition, have been explored to expedite and approximate the inversion of the Hessian matrix, with the aim of making influence functions applicable to deep models. In this paper, we revisit a specific, albeit naive, yet effective approximation method known as TracIn. This method substitutes the inverse of the Hessian matrix with an identity matrix. We provide deeper insights into why this simple approximation method performs well. Furthermore, we extend its applications beyond measuring model utility to include considerations of fairness and robustness. Finally, we enhance TracIn through an ensemble strategy. To validate its effectiveness, we conduct experiments on synthetic data and extensive evaluations on noisy label detection, sample selection for large language model fine-tuning, and defense against adversarial attacks.



## **25. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

cs.CL

29 pages, 6 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15984v1) [paper-pdf](http://arxiv.org/pdf/2405.15984v1)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl



## **26. $$\mathbf{L^2\cdot M = C^2}$$ Large Language Models as Covert Channels... a Systematic Analysis**

cs.CR

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15652v1) [paper-pdf](http://arxiv.org/pdf/2405.15652v1)

**Authors**: Simen Gaure, Stefanos Koffas, Stjepan Picek, Sondre Rønjom

**Abstract**: Large Language Models (LLMs) have gained significant popularity in the last few years due to their performance in diverse tasks such as translation, prediction, or content generation. At the same time, the research community has shown that LLMs are susceptible to various attacks but can also improve the security of diverse systems. However, besides enabling more secure systems, how well do open source LLMs behave as covertext distributions to, e.g., facilitate censorship resistant communication?   In this paper, we explore the capabilities of open-source LLM-based covert channels. We approach this problem from the experimental side by empirically measuring the security vs. capacity of the open-source LLM model (Llama-7B) to assess how well it performs as a covert channel. Although our results indicate that such channels are not likely to achieve high practical bitrates, which depend on message length and model entropy, we also show that the chance for an adversary to detect covert communication is low. To ensure that our results can be used with the least effort as a general reference, we employ a conceptually simple and concise scheme and only assume public models.



## **27. Efficient Adversarial Training in LLMs with Continuous Attacks**

cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15589v1) [paper-pdf](http://arxiv.org/pdf/2405.15589v1)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on four models from different families (Gemma, Phi3, Mistral, Zephyr) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.



## **28. DAGER: Exact Gradient Inversion for Large Language Models**

cs.LG

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15586v1) [paper-pdf](http://arxiv.org/pdf/2405.15586v1)

**Authors**: Ivo Petrov, Dimitar I. Dimitrov, Maximilian Baader, Mark Niklas Müller, Martin Vechev

**Abstract**: Federated learning works by aggregating locally computed gradients from multiple clients, thus enabling collaborative training without sharing private client data. However, prior work has shown that the data can actually be recovered by the server using so-called gradient inversion attacks. While these attacks perform well when applied on images, they are limited in the text domain and only permit approximate reconstruction of small batches and short input sequences. In this work, we propose DAGER, the first algorithm to recover whole batches of input text exactly. DAGER leverages the low-rank structure of self-attention layer gradients and the discrete nature of token embeddings to efficiently check if a given token sequence is part of the client data. We use this check to exactly recover full batches in the honest-but-curious setting without any prior on the data for both encoder- and decoder-based architectures using exhaustive heuristic search and a greedy approach, respectively. We provide an efficient GPU implementation of DAGER and show experimentally that it recovers full batches of size up to 128 on large language models (LLMs), beating prior attacks in speed (20x at same batch size), scalability (10x larger batches), and reconstruction quality (ROUGE-1/2 > 0.99).



## **29. Mosaic Memory: Fuzzy Duplication in Copyright Traps for Large Language Models**

cs.CL

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15523v1) [paper-pdf](http://arxiv.org/pdf/2405.15523v1)

**Authors**: Igor Shilov, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: The immense datasets used to develop Large Language Models (LLMs) often include copyright-protected content, typically without the content creator's consent. Copyright traps have been proposed to be injected into the original content, improving content detectability in newly released LLMs. Traps, however, rely on the exact duplication of a unique text sequence, leaving them vulnerable to commonly deployed data deduplication techniques. We here propose the generation of fuzzy copyright traps, featuring slight modifications across duplication. When injected in the fine-tuning data of a 1.3B LLM, we show fuzzy trap sequences to be memorized nearly as well as exact duplicates. Specifically, the Membership Inference Attack (MIA) ROC AUC only drops from 0.90 to 0.87 when 4 tokens are replaced across the fuzzy duplicates. We also find that selecting replacement positions to minimize the exact overlap between fuzzy duplicates leads to similar memorization, while making fuzzy duplicates highly unlikely to be removed by any deduplication process. Lastly, we argue that the fact that LLMs memorize across fuzzy duplicates challenges the study of LLM memorization relying on naturally occurring duplicates. Indeed, we find that the commonly used training dataset, The Pile, contains significant amounts of fuzzy duplicates. This introduces a previously unexplored confounding factor in post-hoc studies of LLM memorization, and questions the effectiveness of (exact) data deduplication as a privacy protection technique.



## **30. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

cs.CR

18 pages, 13 figures, 4 tables

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.13401v2) [paper-pdf](http://arxiv.org/pdf/2405.13401v2)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.



## **31. Representation noising effectively prevents harmful fine-tuning on LLMs**

cs.CL

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14577v1) [paper-pdf](http://arxiv.org/pdf/2405.14577v1)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that is effective even when attackers have access to the weights and the defender no longer has any control. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the effectiveness of our defence lies in its "depth": the degree to which information about harmful representations is removed across all layers of the LLM.



## **32. Self-playing Adversarial Language Game Enhances LLM Reasoning**

cs.CL

Preprint

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2404.10642v2) [paper-pdf](http://arxiv.org/pdf/2404.10642v2)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by self-play in this adversarial language game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is at https://github.com/Linear95/SPAG.



## **33. Towards Transferable Attacks Against Vision-LLMs in Autonomous Driving with Typography**

cs.CV

12 pages, 5 tables, 5 figures, work in progress

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14169v1) [paper-pdf](http://arxiv.org/pdf/2405.14169v1)

**Authors**: Nhat Chung, Sensen Gao, Tuan-Anh Vu, Jie Zhang, Aishan Liu, Yun Lin, Jin Song Dong, Qing Guo

**Abstract**: Vision-Large-Language-Models (Vision-LLMs) are increasingly being integrated into autonomous driving (AD) systems due to their advanced visual-language reasoning capabilities, targeting the perception, prediction, planning, and control mechanisms. However, Vision-LLMs have demonstrated susceptibilities against various types of adversarial attacks, which would compromise their reliability and safety. To further explore the risk in AD systems and the transferability of practical threats, we propose to leverage typographic attacks against AD systems relying on the decision-making capabilities of Vision-LLMs. Different from the few existing works developing general datasets of typographic attacks, this paper focuses on realistic traffic scenarios where these attacks can be deployed, on their potential effects on the decision-making autonomy, and on the practical ways in which these attacks can be physically presented. To achieve the above goals, we first propose a dataset-agnostic framework for automatically generating false answers that can mislead Vision-LLMs' reasoning. Then, we present a linguistic augmentation scheme that facilitates attacks at image-level and region-level reasoning, and we extend it with attack patterns against multiple reasoning tasks simultaneously. Based on these, we conduct a study on how these attacks can be realized in physical traffic scenarios. Through our empirical study, we evaluate the effectiveness, transferability, and realizability of typographic attacks in traffic scenes. Our findings demonstrate particular harmfulness of the typographic attacks against existing Vision-LLMs (e.g., LLaVA, Qwen-VL, VILA, and Imp), thereby raising community awareness of vulnerabilities when incorporating such models into AD systems. We will release our source code upon acceptance.



## **34. WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation in Query and Response**

cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14023v1) [paper-pdf](http://arxiv.org/pdf/2405.14023v1)

**Authors**: Tianrong Zhang, Bochuan Cao, Yuanpu Cao, Lu Lin, Prasenjit Mitra, Jinghui Chen

**Abstract**: The recent breakthrough in large language models (LLMs) such as ChatGPT has revolutionized production processes at an unprecedented pace. Alongside this progress also comes mounting concerns about LLMs' susceptibility to jailbreaking attacks, which leads to the generation of harmful or unsafe content. While safety alignment measures have been implemented in LLMs to mitigate existing jailbreak attempts and force them to become increasingly complicated, it is still far from perfect. In this paper, we analyze the common pattern of the current safety alignment and show that it is possible to exploit such patterns for jailbreaking attacks by simultaneous obfuscation in queries and responses. Specifically, we propose WordGame attack, which replaces malicious words with word games to break down the adversarial intent of a query and encourage benign content regarding the games to precede the anticipated harmful content in the response, creating a context that is hardly covered by any corpus used for safety alignment. Extensive experiments demonstrate that WordGame attack can break the guardrails of the current leading proprietary and open-source LLMs, including the latest Claude-3, GPT-4, and Llama-3 models. Further ablation studies on such simultaneous obfuscation in query and response provide evidence of the merits of the attack strategy beyond an individual attack.



## **35. Unleashing the Power of Unlabeled Data: A Self-supervised Learning Framework for Cyber Attack Detection in Smart Grids**

cs.LG

9 pages, 5 figures

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13965v1) [paper-pdf](http://arxiv.org/pdf/2405.13965v1)

**Authors**: Hanyu Zeng, Pengfei Zhou, Xin Lou, Zhen Wei Ng, David K. Y. Yau, Marianne Winslett

**Abstract**: Modern power grids are undergoing significant changes driven by information and communication technologies (ICTs), and evolving into smart grids with higher efficiency and lower operation cost. Using ICTs, however, comes with an inevitable side effect that makes the power system more vulnerable to cyber attacks. In this paper, we propose a self-supervised learning-based framework to detect and identify various types of cyber attacks. Different from existing approaches, the proposed framework does not rely on large amounts of well-curated labeled data but makes use of the massive unlabeled data in the wild which are easily accessible. Specifically, the proposed framework adopts the BERT model from the natural language processing domain and learns generalizable and effective representations from the unlabeled sensing data, which capture the distinctive patterns of different attacks. Using the learned representations, together with a very small amount of labeled data, we can train a task-specific classifier to detect various types of cyber attacks. Meanwhile, real-world training datasets are usually imbalanced, i.e., there are only a limited number of data samples containing attacks. In order to cope with such data imbalance, we propose a new loss function, separate mean error (SME), which pays equal attention to the large and small categories to better train the model. Experiment results in a 5-area power grid system with 37 buses demonstrate the superior performance of our framework over existing approaches, especially when a very limited portion of labeled data are available, e.g., as low as 0.002\%. We believe such a framework can be easily adopted to detect a variety of cyber attacks in other power grid scenarios.



## **36. Safety Alignment for Vision Language Models**

cs.CV

23 pages, 15 figures

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13581v1) [paper-pdf](http://arxiv.org/pdf/2405.13581v1)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to an LLMs can realize Vision Language Models (VLMs). However, existing research shows that the visual modality of VLMs is vulnerable, with attackers easily bypassing LLMs' safety alignment through visual modality features to launch attacks. To address this issue, we enhance the existing VLMs' visual modality safety alignment by adding safety modules, including a safety projector, safety tokens, and a safety head, through a two-stage training process, effectively improving the model's defense against risky images. For example, building upon the LLaVA-v1.5 model, we achieve a safety score of 8.26, surpassing the GPT-4V on the Red Teaming Visual Language Models (RTVLM) benchmark. Our method boasts ease of use, high flexibility, and strong controllability, and it enhances safety while having minimal impact on the model's general performance. Moreover, our alignment strategy also uncovers some possible risky content within commonly used open-source multimodal datasets. Our code will be open sourced after the anonymous review.



## **37. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

cs.CR

Camera ready version for GECCO'24 workshop

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2401.15335v2) [paper-pdf](http://arxiv.org/pdf/2401.15335v2)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.



## **38. Information Leakage from Embedding in Large Language Models**

cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.11916v3) [paper-pdf](http://arxiv.org/pdf/2405.11916v3)

**Authors**: Zhipeng Wan, Anda Cheng, Yinggui Wang, Lei Wang

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns regarding data privacy. This study aims to investigate the potential for privacy invasion through input reconstruction attacks, in which a malicious model provider could potentially recover user inputs from embeddings. We first propose two base methods to reconstruct original texts from a model's hidden states. We find that these two methods are effective in attacking the embeddings from shallow layers, but their effectiveness decreases when attacking embeddings from deeper layers. To address this issue, we then present Embed Parrot, a Transformer-based method, to reconstruct input from embeddings in deep layers. Our analysis reveals that Embed Parrot effectively reconstructs original inputs from the hidden states of ChatGLM-6B and Llama2-7B, showcasing stable performance across various token lengths and data distributions. To mitigate the risk of privacy breaches, we introduce a defense mechanism to deter exploitation of the embedding reconstruction process. Our findings emphasize the importance of safeguarding user privacy in distributed learning systems and contribute valuable insights to enhance the security protocols within such environments.



## **39. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.01218v3) [paper-pdf](http://arxiv.org/pdf/2403.01218v3)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their "U-MIA" counterparts). We propose a categorization of existing U-MIAs into "population U-MIAs", where the same attacker is instantiated for all examples, and "per-example U-MIAs", where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.



## **40. Generative AI and Large Language Models for Cyber Security: All Insights You Need**

cs.CR

50 pages, 8 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12750v1) [paper-pdf](http://arxiv.org/pdf/2405.12750v1)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.



## **41. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

cs.CR

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.13077v1) [paper-pdf](http://arxiv.org/pdf/2405.13077v1)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find IRIS achieves jailbreak success rates of 98% on GPT-4 and 92% on GPT-4 Turbo in under 7 queries. It significantly outperforms prior approaches in automatic, black-box and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.



## **42. Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation**

cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.13068v1) [paper-pdf](http://arxiv.org/pdf/2405.13068v1)

**Authors**: Yuxi Li, Yi Liu, Yuekang Li, Ling Shi, Gelei Deng, Shengquan Chen, Kailong Wang

**Abstract**: Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JailMine, an innovative token-level manipulation approach that addresses these limitations effectively. JailMine employs an automated "mining" process to elicit malicious responses from LLMs by strategically selecting affirmative outputs and iteratively reducing the likelihood of rejection. Through rigorous testing across multiple well-known LLMs and datasets, we demonstrate JailMine's effectiveness and efficiency, achieving a significant average reduction of 86% in time consumed while maintaining high success rates averaging 95%, even in the face of evolving defensive strategies. Our work contributes to the ongoing effort to assess and mitigate the vulnerability of LLMs to jailbreaking attacks, underscoring the importance of continued vigilance and proactive measures to enhance the security and reliability of these powerful language models.



## **43. Special Characters Attack: Toward Scalable Training Data Extraction From Large Language Models**

cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.05990v2) [paper-pdf](http://arxiv.org/pdf/2405.05990v2)

**Authors**: Yang Bai, Ge Pei, Jindong Gu, Yong Yang, Xingjun Ma

**Abstract**: Large language models (LLMs) have achieved remarkable performance on a wide range of tasks. However, recent studies have shown that LLMs can memorize training data and simple repeated tokens can trick the model to leak the data. In this paper, we take a step further and show that certain special characters or their combinations with English letters are stronger memory triggers, leading to more severe data leakage. The intuition is that, since LLMs are trained with massive data that contains a substantial amount of special characters (e.g. structural symbols {, } of JSON files, and @, # in emails and online posts), the model may memorize the co-occurrence between these special characters and the raw texts. This motivates us to propose a simple but effective Special Characters Attack (SCA) to induce training data leakage. Our experiments verify the high effectiveness of SCA against state-of-the-art LLMs: they can leak diverse training data, such as code corpus, web pages, and personally identifiable information, and sometimes generate non-stop outputs as a byproduct. We further show that the composition of the training data corpus can be revealed by inspecting the leaked data -- one crucial piece of information for pre-training high-performance LLMs. Our work can help understand the sensitivity of LLMs to special characters and identify potential areas for improvement.



## **44. Data Contamination Calibration for Black-box LLMs**

cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11930v1) [paper-pdf](http://arxiv.org/pdf/2405.11930v1)

**Authors**: Wentao Ye, Jiaqi Hu, Liyao Li, Haobo Wang, Gang Chen, Junbo Zhao

**Abstract**: The rapid advancements of Large Language Models (LLMs) tightly associate with the expansion of the training data size. However, the unchecked ultra-large-scale training sets introduce a series of potential risks like data contamination, i.e. the benchmark data is used for training. In this work, we propose a holistic method named Polarized Augment Calibration (PAC) along with a new to-be-released dataset to detect the contaminated data and diminish the contamination effect. PAC extends the popular MIA (Membership Inference Attack) -- from machine learning community -- by forming a more global target at detecting training data to Clarify invisible training data. As a pioneering work, PAC is very much plug-and-play that can be integrated with most (if not all) current white- and black-box LLMs. By extensive experiments, PAC outperforms existing methods by at least 4.5%, towards data contamination detection on more 4 dataset formats, with more than 10 base LLMs. Besides, our application in real-world scenarios highlights the prominent presence of contamination and related issues.



## **45. A Semantic Invariant Robust Watermark for Large Language Models**

cs.CR

ICLR2024, 21 pages, 10 figures, 6 tables

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2310.06356v3) [paper-pdf](http://arxiv.org/pdf/2310.06356v3)

**Authors**: Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen

**Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at \href{https://github.com/THU-BPM/Robust_Watermark}{https://github.com/THU-BPM/Robust\_Watermark}. Additionally, our algorithm could also be accessed through MarkLLM \citep{pan2024markllm} \footnote{https://github.com/THU-BPM/MarkLLM}.



## **46. Measuring Impacts of Poisoning on Model Parameters and Embeddings for Large Language Models of Code**

cs.SE

This work has been accepted at the 1st ACM International Conference  on AI-powered Software (AIware), co-located with the ACM International  Conference on the Foundations of Software Engineering (FSE) 2024, Porto de  Galinhas, Brazil. arXiv admin note: substantial text overlap with  arXiv:2402.12936

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11466v1) [paper-pdf](http://arxiv.org/pdf/2405.11466v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have revolutionized software development practices, yet concerns about their safety have arisen, particularly regarding hidden backdoors, aka trojans. Backdoor attacks involve the insertion of triggers into training data, allowing attackers to manipulate the behavior of the model maliciously. In this paper, we focus on analyzing the model parameters to detect potential backdoor signals in code models. Specifically, we examine attention weights and biases, and context embeddings of the clean and poisoned CodeBERT and CodeT5 models. Our results suggest noticeable patterns in context embeddings of poisoned samples for both the poisoned models; however, attention weights and biases do not show any significant differences. This work contributes to ongoing efforts in white-box detection of backdoor signals in LLMs of code through the analysis of parameters and embeddings.



## **47. Revisiting the Robust Generalization of Adversarial Prompt Tuning**

cs.CV

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11154v1) [paper-pdf](http://arxiv.org/pdf/2405.11154v1)

**Authors**: Fan Yang, Mingxuan Xia, Sangzhou Xia, Chicheng Ma, Hui Hui

**Abstract**: Understanding the vulnerability of large-scale pre-trained vision-language models like CLIP against adversarial attacks is key to ensuring zero-shot generalization capacity on various downstream tasks. State-of-the-art defense mechanisms generally adopt prompt learning strategies for adversarial fine-tuning to improve the adversarial robustness of the pre-trained model while keeping the efficiency of adapting to downstream tasks. Such a setup leads to the problem of over-fitting which impedes further improvement of the model's generalization capacity on both clean and adversarial examples. In this work, we propose an adaptive Consistency-guided Adversarial Prompt Tuning (i.e., CAPT) framework that utilizes multi-modal prompt learning to enhance the alignment of image and text features for adversarial examples and leverage the strong generalization of pre-trained CLIP to guide the model-enhancing its robust generalization on adversarial examples while maintaining its accuracy on clean ones. We also design a novel adaptive consistency objective function to balance the consistency of adversarial inputs and clean inputs between the fine-tuning model and the pre-trained model. We conduct extensive experiments across 14 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show the superiority of CAPT over other state-of-the-art adaption methods. CAPT demonstrated excellent performance in terms of the in-distribution performance and the generalization under input distribution shift and across datasets.



## **48. A Comprehensive Study of Jailbreak Attack versus Defense for Large Language Models**

cs.CR

18 pages, 9 figures, Accepted in ACL 2024

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2402.13457v2) [paper-pdf](http://arxiv.org/pdf/2402.13457v2)

**Authors**: Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, Stjepan Picek

**Abstract**: Large Language Models (LLMS) have increasingly become central to generating content with potential societal impacts. Notably, these models have demonstrated capabilities for generating content that could be deemed harmful. To mitigate these risks, researchers have adopted safety training techniques to align model outputs with societal values to curb the generation of malicious content. However, the phenomenon of "jailbreaking", where carefully crafted prompts elicit harmful responses from models, persists as a significant challenge. This research conducts a comprehensive analysis of existing studies on jailbreaking LLMs and their defense techniques. We meticulously investigate nine attack techniques and seven defense techniques applied across three distinct language models: Vicuna, LLama, and GPT-3.5 Turbo. We aim to evaluate the effectiveness of these attack and defense techniques. Our findings reveal that existing white-box attacks underperform compared to universal techniques and that including special tokens in the input significantly affects the likelihood of successful attacks. This research highlights the need to concentrate on the security facets of LLMs. Additionally, we contribute to the field by releasing our datasets and testing framework, aiming to foster further research into LLM security. We believe these contributions will facilitate the exploration of security measures within this domain.



## **49. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

cs.CV

15 pages

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.10529v1) [paper-pdf](http://arxiv.org/pdf/2405.10529v1)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.



## **50. Large Language Models in Wireless Application Design: In-Context Learning-enhanced Automatic Network Intrusion Detection**

cs.LG

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.11002v1) [paper-pdf](http://arxiv.org/pdf/2405.11002v1)

**Authors**: Han Zhang, Akram Bin Sediq, Ali Afana, Melike Erol-Kantarci

**Abstract**: Large language models (LLMs), especially generative pre-trained transformers (GPTs), have recently demonstrated outstanding ability in information comprehension and problem-solving. This has motivated many studies in applying LLMs to wireless communication networks. In this paper, we propose a pre-trained LLM-empowered framework to perform fully automatic network intrusion detection. Three in-context learning methods are designed and compared to enhance the performance of LLMs. With experiments on a real network intrusion detection dataset, in-context learning proves to be highly beneficial in improving the task processing performance in a way that no further training or fine-tuning of LLMs is required. We show that for GPT-4, testing accuracy and F1-Score can be improved by 90%. Moreover, pre-trained LLMs demonstrate big potential in performing wireless communication-related tasks. Specifically, the proposed framework can reach an accuracy and F1-Score of over 95% on different types of attacks with GPT-4 using only 10 in-context learning examples.



