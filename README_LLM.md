# Latest Large Language Model Attack Papers
**update at 2024-10-14 09:43:09**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. On the Adversarial Transferability of Generalized "Skip Connections"**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08950v1) [paper-pdf](http://arxiv.org/pdf/2410.08950v1)

**Authors**: Yisen Wang, Yichuan Mo, Dongxian Wu, Mingjie Li, Xingjun Ma, Zhouchen Lin

**Abstract**: Skip connection is an essential ingredient for modern deep models to be deeper and more powerful. Despite their huge success in normal scenarios (state-of-the-art classification performance on natural examples), we investigate and identify an interesting property of skip connections under adversarial scenarios, namely, the use of skip connections allows easier generation of highly transferable adversarial examples. Specifically, in ResNet-like models (with skip connections), we find that using more gradients from the skip connections rather than the residual modules according to a decay factor during backpropagation allows one to craft adversarial examples with high transferability. The above method is termed as Skip Gradient Method (SGM). Although starting from ResNet-like models in vision domains, we further extend SGM to more advanced architectures, including Vision Transformers (ViTs) and models with length-varying paths and other domains, i.e. natural language processing. We conduct comprehensive transfer attacks against various models including ResNets, Transformers, Inceptions, Neural Architecture Search, and Large Language Models (LLMs). We show that employing SGM can greatly improve the transferability of crafted attacks in almost all cases. Furthermore, considering the big complexity for practical use, we further demonstrate that SGM can even improve the transferability on ensembles of models or targeted attacks and the stealthiness against current defenses. At last, we provide theoretical explanations and empirical insights on how SGM works. Our findings not only motivate new adversarial research into the architectural characteristics of models but also open up further challenges for secure model architecture design. Our code is available at https://github.com/mo666666/SGM.



## **2. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08827v1) [paper-pdf](http://arxiv.org/pdf/2410.08827v1)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.



## **3. PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning**

cs.CR

Tingchen Fu and Fazl Barez are core research contributors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08811v1) [paper-pdf](http://arxiv.org/pdf/2410.08811v1)

**Authors**: Tingchen Fu, Mrinank Sharma, Philip Torr, Shay B. Cohen, David Krueger, Fazl Barez

**Abstract**: Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 21 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.



## **4. F2A: An Innovative Approach for Prompt Injection by Utilizing Feign Security Detection Agents**

cs.CR

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08776v1) [paper-pdf](http://arxiv.org/pdf/2410.08776v1)

**Authors**: Yupeng Ren

**Abstract**: With the rapid development of Large Language Models (LLMs), numerous mature applications of LLMs have emerged in the field of content safety detection. However, we have found that LLMs exhibit blind trust in safety detection agents. The general LLMs can be compromised by hackers with this vulnerability. Hence, this paper proposed an attack named Feign Agent Attack (F2A).Through such malicious forgery methods, adding fake safety detection results into the prompt, the defense mechanism of LLMs can be bypassed, thereby obtaining harmful content and hijacking the normal conversation.Continually, a series of experiments were conducted. In these experiments, the hijacking capability of F2A on LLMs was analyzed and demonstrated, exploring the fundamental reasons why LLMs blindly trust safety detection results. The experiments involved various scenarios where fake safety detection results were injected into prompts, and the responses were closely monitored to understand the extent of the vulnerability. Also, this paper provided a reasonable solution to this attack, emphasizing that it is important for LLMs to critically evaluate the results of augmented agents to prevent the generating harmful content. By doing so, the reliability and security can be significantly improved, protecting the LLMs from F2A.



## **5. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

cs.CR

arXiv admin note: text overlap with arXiv:2403.04783 by other authors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08660v1) [paper-pdf](http://arxiv.org/pdf/2410.08660v1)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.



## **6. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2405.19360v3) [paper-pdf](http://arxiv.org/pdf/2405.19360v3)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.



## **7. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

cs.CL

12 pages, 9 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.21659v3) [paper-pdf](http://arxiv.org/pdf/2407.21659v3)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.



## **8. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06186v2) [paper-pdf](http://arxiv.org/pdf/2410.06186v2)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.



## **9. APOLLO: A GPT-based tool to detect phishing emails and generate explanations that warn users**

cs.HC

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07997v1) [paper-pdf](http://arxiv.org/pdf/2410.07997v1)

**Authors**: Giuseppe Desolda, Francesco Greco, Luca Viganò

**Abstract**: Phishing is one of the most prolific cybercriminal activities, with attacks becoming increasingly sophisticated. It is, therefore, imperative to explore novel technologies to improve user protection across both technical and human dimensions. Large Language Models (LLMs) offer significant promise for text processing in various domains, but their use for defense against phishing attacks still remains scarcely explored. In this paper, we present APOLLO, a tool based on OpenAI's GPT-4o to detect phishing emails and generate explanation messages to users about why a specific email is dangerous, thus improving their decision-making capabilities. We have evaluated the performance of APOLLO in classifying phishing emails; the results show that the LLM models have exemplary capabilities in classifying phishing emails (97 percent accuracy in the case of GPT-4o) and that this performance can be further improved by integrating data from third-party services, resulting in a near-perfect classification rate (99 percent accuracy). To assess the perception of the explanations generated by this tool, we also conducted a study with 20 participants, comparing four different explanations presented as phishing warnings. We compared the LLM-generated explanations to four baselines: a manually crafted warning, and warnings from Chrome, Firefox, and Edge browsers. The results show that not only the LLM-generated explanations were perceived as high quality, but also that they can be more understandable, interesting, and trustworthy than the baselines. These findings suggest that using LLMs as a defense against phishing is a very promising approach, with APOLLO representing a proof of concept in this research direction.



## **10. Towards Assurance of LLM Adversarial Robustness using Ontology-Driven Argumentation**

cs.AI

To be published in xAI 2024, late-breaking track

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07962v1) [paper-pdf](http://arxiv.org/pdf/2410.07962v1)

**Authors**: Tomas Bueno Momcilovic, Beat Buesser, Giulio Zizzo, Mark Purcell, Dian Balta

**Abstract**: Despite the impressive adaptability of large language models (LLMs), challenges remain in ensuring their security, transparency, and interpretability. Given their susceptibility to adversarial attacks, LLMs need to be defended with an evolving combination of adversarial training and guardrails. However, managing the implicit and heterogeneous knowledge for continuously assuring robustness is difficult. We introduce a novel approach for assurance of the adversarial robustness of LLMs based on formal argumentation. Using ontologies for formalization, we structure state-of-the-art attacks and defenses, facilitating the creation of a human-readable assurance case, and a machine-readable representation. We demonstrate its application with examples in English language and code translation tasks, and provide implications for theory and practice, by targeting engineers, data scientists, users, and auditors.



## **11. Protecting Your LLMs with Information Bottleneck**

cs.CL

Accepted by Neural Information Processing Systems (NeurIPS 2024)

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2404.13968v3) [paper-pdf](http://arxiv.org/pdf/2404.13968v3)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **12. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.02890v2) [paper-pdf](http://arxiv.org/pdf/2410.02890v2)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.



## **13. Mind Your Questions! Towards Backdoor Attacks on Text-to-Visualization Models**

cs.CR

11 pages, 4 figures

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06782v2) [paper-pdf](http://arxiv.org/pdf/2410.06782v2)

**Authors**: Shuaimin Li, Yuanfeng Song, Xuanang Chen, Anni Peng, Zhuoyue Wan, Chen Jason Zhang, Raymond Chi-Wing Wong

**Abstract**: Text-to-visualization (text-to-vis) models have become valuable tools in the era of big data, enabling users to generate data visualizations and make informed decisions through natural language queries (NLQs). Despite their widespread application, the security vulnerabilities of these models have been largely overlooked. To address this gap, we propose VisPoison, a novel framework designed to identify these vulnerabilities of current text-to-vis models systematically. VisPoison introduces two types of triggers that activate three distinct backdoor attacks, potentially leading to data exposure, misleading visualizations, or denial-of-service (DoS) incidents. The framework features both proactive and passive attack mechanisms: proactive attacks leverage rare-word triggers to access confidential data, while passive attacks, triggered unintentionally by users, exploit a first-word trigger method, causing errors or DoS events in visualizations. Through extensive experiments on both trainable and in-context learning (ICL)-based text-to-vis models, \textit{VisPoison} achieves attack success rates of over 90\%, highlighting the security problem of current text-to-vis models. Additionally, we explore two types of defense mechanisms against these attacks, but the results show that existing countermeasures are insufficient, underscoring the pressing need for more robust security solutions in text-to-vis systems.



## **14. Detecting Training Data of Large Language Models via Expectation Maximization**

cs.CL

14 pages

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07582v1) [paper-pdf](http://arxiv.org/pdf/2410.07582v1)

**Authors**: Gyuwan Kim, Yang Li, Evangelia Spiliopoulou, Jie Ma, Miguel Ballesteros, William Yang Wang

**Abstract**: The widespread deployment of large language models (LLMs) has led to impressive advancements, yet information about their training data, a critical factor in their performance, remains undisclosed. Membership inference attacks (MIAs) aim to determine whether a specific instance was part of a target model's training data. MIAs can offer insights into LLM outputs and help detect and address concerns such as data contamination and compliance with privacy and copyright standards. However, applying MIAs to LLMs presents unique challenges due to the massive scale of pre-training data and the ambiguous nature of membership. Additionally, creating appropriate benchmarks to evaluate MIA methods is not straightforward, as training and test data distributions are often unknown. In this paper, we introduce EM-MIA, a novel MIA method for LLMs that iteratively refines membership scores and prefix scores via an expectation-maximization algorithm, leveraging the duality that the estimates of these scores can be improved by each other. Membership scores and prefix scores assess how each instance is likely to be a member and discriminative as a prefix, respectively. Our method achieves state-of-the-art results on the WikiMIA dataset. To further evaluate EM-MIA, we present OLMoMIA, a benchmark built from OLMo resources, which allows us to control the difficulty of MIA tasks with varying degrees of overlap between training and test data distributions. We believe that EM-MIA serves as a robust MIA method for LLMs and that OLMoMIA provides a valuable resource for comprehensively evaluating MIA approaches, thereby driving future research in this critical area.



## **15. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07163v1) [paper-pdf](http://arxiv.org/pdf/2410.07163v1)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities (e.g., copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty. Given that, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.



## **16. Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning**

cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2401.05949v6) [paper-pdf](http://arxiv.org/pdf/2401.05949v6)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we design a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning demonstration prompts, which can make models behave in alignment with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 180B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models.



## **17. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems**

cs.MA

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07283v1) [paper-pdf](http://arxiv.org/pdf/2410.07283v1)

**Authors**: Donghyun Lee, Mo Tiwari

**Abstract**: As Large Language Models (LLMs) grow increasingly powerful, multi-agent systems are becoming more prevalent in modern AI applications. Most safety research, however, has focused on vulnerabilities in single-agent LLMs. These include prompt injection attacks, where malicious prompts embedded in external content trick the LLM into executing unintended or harmful actions, compromising the victim's application. In this paper, we reveal a more dangerous vector: LLM-to-LLM prompt injection within multi-agent systems. We introduce Prompt Infection, a novel attack where malicious prompts self-replicate across interconnected agents, behaving much like a computer virus. This attack poses severe threats, including data theft, scams, misinformation, and system-wide disruption, all while propagating silently through the system. Our extensive experiments demonstrate that multi-agent systems are highly susceptible, even when agents do not publicly share all communications. To address this, we propose LLM Tagging, a defense mechanism that, when combined with existing safeguards, significantly mitigates infection spread. This work underscores the urgent need for advanced security measures as multi-agent LLM systems become more widely adopted.



## **18. Break the Visual Perception: Adversarial Attacks Targeting Encoded Visual Tokens of Large Vision-Language Models**

cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06699v1) [paper-pdf](http://arxiv.org/pdf/2410.06699v1)

**Authors**: Yubo Wang, Chaohu Liu, Yanqiu Qu, Haoyu Cao, Deqiang Jiang, Linli Xu

**Abstract**: Large vision-language models (LVLMs) integrate visual information into large language models, showcasing remarkable multi-modal conversational capabilities. However, the visual modules introduces new challenges in terms of robustness for LVLMs, as attackers can craft adversarial images that are visually clean but may mislead the model to generate incorrect answers. In general, LVLMs rely on vision encoders to transform images into visual tokens, which are crucial for the language models to perceive image contents effectively. Therefore, we are curious about one question: Can LVLMs still generate correct responses when the encoded visual tokens are attacked and disrupting the visual information? To this end, we propose a non-targeted attack method referred to as VT-Attack (Visual Tokens Attack), which constructs adversarial examples from multiple perspectives, with the goal of comprehensively disrupting feature representations and inherent relationships as well as the semantic properties of visual tokens output by image encoders. Using only access to the image encoder in the proposed attack, the generated adversarial examples exhibit transferability across diverse LVLMs utilizing the same image encoder and generality across different tasks. Extensive experiments validate the superior attack performance of the VT-Attack over baseline methods, demonstrating its effectiveness in attacking LVLMs with image encoders, which in turn can provide guidance on the robustness of LVLMs, particularly in terms of the stability of the visual feature space.



## **19. FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services**

cs.IR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.04927v2) [paper-pdf](http://arxiv.org/pdf/2410.04927v2)

**Authors**: Wei Yuan, Chaoqun Yang, Guanhua Ye, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated sequential recommendation (FedSeqRec) has gained growing attention due to its ability to protect user privacy. Unfortunately, the performance of FedSeqRec is still unsatisfactory because the models used in FedSeqRec have to be lightweight to accommodate communication bandwidth and clients' on-device computational resource constraints. Recently, large language models (LLMs) have exhibited strong transferable and generalized language understanding abilities and therefore, in the NLP area, many downstream tasks now utilize LLMs as a service to achieve superior performance without constructing complex models. Inspired by this successful practice, we propose a generic FedSeqRec framework, FELLAS, which aims to enhance FedSeqRec by utilizing LLMs as an external service. Specifically, FELLAS employs an LLM server to provide both item-level and sequence-level representation assistance. The item-level representation service is queried by the central server to enrich the original ID-based item embedding with textual information, while the sequence-level representation service is accessed by each client. However, invoking the sequence-level representation service requires clients to send sequences to the external LLM server. To safeguard privacy, we implement dx-privacy satisfied sequence perturbation, which protects clients' sensitive data with guarantees. Additionally, a contrastive learning-based method is designed to transfer knowledge from the noisy sequence representation to clients' sequential recommendation models. Furthermore, to empirically validate the privacy protection capability of FELLAS, we propose two interacted item inference attacks. Extensive experiments conducted on three datasets with two widely used sequential recommendation models demonstrate the effectiveness and privacy-preserving capability of FELLAS.



## **20. Signal Watermark on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06545v1) [paper-pdf](http://arxiv.org/pdf/2410.06545v1)

**Authors**: Zhenyu Xu, Victor S. Sheng

**Abstract**: As Large Language Models (LLMs) become increasingly sophisticated, they raise significant security concerns, including the creation of fake news and academic misuse. Most detectors for identifying model-generated text are limited by their reliance on variance in perplexity and burstiness, and they require substantial computational resources. In this paper, we proposed a watermarking method embedding a specific watermark into the text during its generation by LLMs, based on a pre-defined signal pattern. This technique not only ensures the watermark's invisibility to humans but also maintains the quality and grammatical integrity of model-generated text. We utilize LLMs and Fast Fourier Transform (FFT) for token probability computation and detection of the signal watermark. The unique application of signal processing principles within the realm of text generation by LLMs allows for subtle yet effective embedding of watermarks, which do not compromise the quality or coherence of the generated text. Our method has been empirically validated across multiple LLMs, consistently maintaining high detection accuracy, even with variations in temperature settings during text generation. In the experiment of distinguishing between human-written and watermarked text, our method achieved an AUROC score of 0.97, significantly outperforming existing methods like GPTZero, which scored 0.64. The watermark's resilience to various attacking scenarios further confirms its robustness, addressing significant challenges in model-generated text authentication.



## **21. WAPITI: A Watermark for Finetuned Open-Source LLMs**

cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06467v1) [paper-pdf](http://arxiv.org/pdf/2410.06467v1)

**Authors**: Lingjie Chen, Ruizhong Qiu, Siyu Yuan, Zhining Liu, Tianxin Wei, Hyunsik Yoo, Zhichen Zeng, Deqing Yang, Hanghang Tong

**Abstract**: Watermarking of large language models (LLMs) generation embeds an imperceptible statistical pattern within texts, making it algorithmically detectable. Watermarking is a promising method for addressing potential harm and biases from LLMs, as it enables traceability, accountability, and detection of manipulated content, helping to mitigate unintended consequences. However, for open-source models, watermarking faces two major challenges: (i) incompatibility with fine-tuned models, and (ii) vulnerability to fine-tuning attacks. In this work, we propose WAPITI, a new method that transfers watermarking from base models to fine-tuned models through parameter integration. To the best of our knowledge, we propose the first watermark for fine-tuned open-source LLMs that preserves their fine-tuned capabilities. Furthermore, our approach offers an effective defense against fine-tuning attacks. We test our method on various model architectures and watermarking strategies. Results demonstrate that our method can successfully inject watermarks and is highly compatible with fine-tuned models. Additionally, we offer an in-depth analysis of how parameter editing influences the watermark strength and overall capabilities of the resulting models.



## **22. Hallucinating AI Hijacking Attack: Large Language Models and Malicious Code Recommenders**

cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06462v1) [paper-pdf](http://arxiv.org/pdf/2410.06462v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: The research builds and evaluates the adversarial potential to introduce copied code or hallucinated AI recommendations for malicious code in popular code repositories. While foundational large language models (LLMs) from OpenAI, Google, and Anthropic guard against both harmful behaviors and toxic strings, previous work on math solutions that embed harmful prompts demonstrate that the guardrails may differ between expert contexts. These loopholes would appear in mixture of expert's models when the context of the question changes and may offer fewer malicious training examples to filter toxic comments or recommended offensive actions. The present work demonstrates that foundational models may refuse to propose destructive actions correctly when prompted overtly but may unfortunately drop their guard when presented with a sudden change of context, like solving a computer programming challenge. We show empirical examples with trojan-hosting repositories like GitHub, NPM, NuGet, and popular content delivery networks (CDN) like jsDelivr which amplify the attack surface. In the LLM's directives to be helpful, example recommendations propose application programming interface (API) endpoints which a determined domain-squatter could acquire and setup attack mobile infrastructure that triggers from the naively copied code. We compare this attack to previous work on context-shifting and contrast the attack surface as a novel version of "living off the land" attacks in the malware literature. In the latter case, foundational language models can hijack otherwise innocent user prompts to recommend actions that violate their owners' safety policies when posed directly without the accompanying coding support request.



## **23. Evaluating and Safeguarding the Adversarial Robustness of Retrieval-Based In-Context Learning**

cs.CL

COLM 2024, 31 pages, 6 figures

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.15984v4) [paper-pdf](http://arxiv.org/pdf/2405.15984v4)

**Authors**: Simon Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl



## **24. Training-free LLM-generated Text Detection by Mining Token Probability Sequences**

cs.CL

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2410.06072v1) [paper-pdf](http://arxiv.org/pdf/2410.06072v1)

**Authors**: Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, Fei Wu

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in generating high-quality texts across diverse domains. However, the potential misuse of LLMs has raised significant concerns, underscoring the urgent need for reliable detection of LLM-generated texts. Conventional training-based detectors often struggle with generalization, particularly in cross-domain and cross-model scenarios. In contrast, training-free methods, which focus on inherent discrepancies through carefully designed statistical features, offer improved generalization and interpretability. Despite this, existing training-free detection methods typically rely on global text sequence statistics, neglecting the modeling of local discriminative features, thereby limiting their detection efficacy. In this work, we introduce a novel training-free detector, termed \textbf{Lastde} that synergizes local and global statistics for enhanced detection. For the first time, we introduce time series analysis to LLM-generated text detection, capturing the temporal dynamics of token probability sequences. By integrating these local statistics with global ones, our detector reveals significant disparities between human and LLM-generated texts. We also propose an efficient alternative, \textbf{Lastde++} to enable real-time detection. Extensive experiments on six datasets involving cross-domain, cross-model, and cross-lingual detection scenarios, under both white-box and black-box settings, demonstrated that our method consistently achieves state-of-the-art performance. Furthermore, our approach exhibits greater robustness against paraphrasing attacks compared to existing baseline methods.



## **25. Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs**

cs.CR

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2409.14866v2) [paper-pdf](http://arxiv.org/pdf/2409.14866v2)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates, our method starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks.   We evaluated our method on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, our method achieves attack success rates of over 90%,80% and 74%, respectively, exceeding existing baselines by more than 60%. Additionally, our method can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, our method can achieve over 78% attack success rate even with 100 tokens. Moreover, our method demonstrates transferability and is robust to state-of-the-art defenses. We will open-source our codes upon publication.



## **26. You Know What I'm Saying: Jailbreak Attack via Implicit Reference**

cs.CL

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2410.03857v2) [paper-pdf](http://arxiv.org/pdf/2410.03857v2)

**Authors**: Tianyu Wu, Lingrui Mei, Ruibin Yuan, Lujun Li, Wei Xue, Yike Guo

**Abstract**: While recent advancements in large language model (LLM) alignment have enabled the effective identification of malicious objectives involving scene nesting and keyword rewriting, our study reveals that these methods remain inadequate at detecting malicious objectives expressed through context within nested harmless objectives. This study identifies a previously overlooked vulnerability, which we term Attack via Implicit Reference (AIR). AIR decomposes a malicious objective into permissible objectives and links them through implicit references within the context. This method employs multiple related harmless objectives to generate malicious content without triggering refusal responses, thereby effectively bypassing existing detection techniques.Our experiments demonstrate AIR's effectiveness across state-of-the-art LLMs, achieving an attack success rate (ASR) exceeding 90% on most models, including GPT-4o, Claude-3.5-Sonnet, and Qwen-2-72B. Notably, we observe an inverse scaling phenomenon, where larger models are more vulnerable to this attack method. These findings underscore the urgent need for defense mechanisms capable of understanding and preventing contextual attacks. Furthermore, we introduce a cross-model attack strategy that leverages less secure models to generate malicious contexts, thereby further increasing the ASR when targeting other models.Our code and jailbreak artifacts can be found at https://github.com/Lucas-TY/llm_Implicit_reference.



## **27. Partially Recentralization Softmax Loss for Vision-Language Models Robustness**

cs.CL

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2402.03627v2) [paper-pdf](http://arxiv.org/pdf/2402.03627v2)

**Authors**: Hao Wang, Jinzhe Jiang, Xin Zhang, Chen Li

**Abstract**: As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after this paper is accepted



## **28. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

cs.CL

18 pages

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.18111v3) [paper-pdf](http://arxiv.org/pdf/2405.18111v3)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, since today's Internet is flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with an Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent through adversarially tuning the agents for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to the state-of-the-art baselines.



## **29. Aligning LLMs to Be Robust Against Prompt Injection**

cs.CR

Key words: prompt injection defense, LLM security, LLM-integrated  applications. Alignment training makes LLMs robust against even the strongest  prompt injection attacks

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05451v1) [paper-pdf](http://arxiv.org/pdf/2410.05451v1)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be carefully crafted and injected into external data sources to override the user's intended instruction and instead execute a malicious instruction. Prompt injection attacks constitute a major threat to LLM security, making the design and implementation of practical countermeasures of paramount importance. To this end, we show that alignment can be a powerful tool to make LLMs more robust against prompt injection. Our method -- SecAlign -- first builds an alignment dataset by simulating prompt injection attacks and constructing pairs of desirable and undesirable responses. Then, we apply existing alignment techniques to fine-tune the LLM to be robust against these simulated attacks. Our experiments show that SecAlign robustifies the LLM substantially with a negligible hurt on model utility. Moreover, SecAlign's protection generalizes to strong attacks unseen in training. Specifically, the success rate of state-of-the-art GCG-based prompt injections drops from 56% to 2% in Mistral-7B after our alignment process. Our code is released at https://github.com/facebookresearch/SecAlign



## **30. $$\mathbf{L^2\cdot M = C^2}$$ Large Language Models are Covert Channels**

cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2405.15652v2) [paper-pdf](http://arxiv.org/pdf/2405.15652v2)

**Authors**: Simen Gaure, Stefanos Koffas, Stjepan Picek, Sondre Rønjom

**Abstract**: Large Language Models (LLMs) have gained significant popularity recently. LLMs are susceptible to various attacks but can also improve the security of diverse systems. However, besides enabling more secure systems, how well do open source LLMs behave as covertext distributions to, e.g., facilitate censorship-resistant communication? In this paper, we explore open-source LLM-based covert channels. We empirically measure the security vs. capacity of an open-source LLM model (Llama-7B) to assess its performance as a covert channel. Although our results indicate that such channels are not likely to achieve high practical bitrates, we also show that the chance for an adversary to detect covert communication is low. To ensure our results can be used with the least effort as a general reference, we employ a conceptually simple and concise scheme and only assume public models.



## **31. Representation noising effectively prevents harmful fine-tuning on LLMs**

cs.CL

Published in NeurIPs 2024

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2405.14577v2) [paper-pdf](http://arxiv.org/pdf/2405.14577v2)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that is effective even when attackers have access to the weights. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process as long as they are drawn from the same distribution of the attack set. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the effectiveness of our defence lies in its "depth": the degree to which information about harmful representations is removed across all layers of the LLM.



## **32. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2312.03853v5) [paper-pdf](http://arxiv.org/pdf/2312.03853v5)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbots. Safety mechanisms are implemented to prevent improper responses from these chatbots. In this work, we bypass these measures for ChatGPT and Gemini by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information in both ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, showing that both chatbots are vulnerable to this attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **33. Reconstruct Your Previous Conversations! Comprehensively Investigating Privacy Leakage Risks in Conversations with GPT Models**

cs.CR

Accepted in EMNLP 2024. 14 pages, 10 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2402.02987v2) [paper-pdf](http://arxiv.org/pdf/2402.02987v2)

**Authors**: Junjie Chu, Zeyang Sha, Michael Backes, Yang Zhang

**Abstract**: Significant advancements have recently been made in large language models represented by GPT models. Users frequently have multi-round private conversations with cloud-hosted GPT models for task optimization. Yet, this operational paradigm introduces additional attack surfaces, particularly in custom GPTs and hijacked chat sessions. In this paper, we introduce a straightforward yet potent Conversation Reconstruction Attack. This attack targets the contents of previous conversations between GPT models and benign users, i.e., the benign users' input contents during their interaction with GPT models. The adversary could induce GPT models to leak such contents by querying them with designed malicious prompts. Our comprehensive examination of privacy risks during the interactions with GPT models under this attack reveals GPT-4's considerable resilience. We present two advanced attacks targeting improved reconstruction of past conversations, demonstrating significant privacy leakage across all models under these advanced techniques. Evaluating various defense mechanisms, we find them ineffective against these attacks. Our findings highlight the ease with which privacy can be compromised in interactions with GPT models, urging the community to safeguard against potential abuses of these models' capabilities.



## **34. AnyAttack: Towards Large-scale Self-supervised Generation of Targeted Adversarial Examples for Vision-Language Models**

cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05346v1) [paper-pdf](http://arxiv.org/pdf/2410.05346v1)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks, particularly targeted adversarial images that manipulate the model to generate harmful content specified by the adversary. Current attack methods rely on predefined target labels to create targeted adversarial attacks, which limits their scalability and applicability for large-scale robustness evaluations. In this paper, we propose AnyAttack, a self-supervised framework that generates targeted adversarial images for VLMs without label supervision, allowing any image to serve as a target for the attack. To address the limitation of existing methods that require label supervision, we introduce a contrastive loss that trains a generator on a large-scale unlabeled image dataset, LAION-400M dataset, for generating targeted adversarial noise. This large-scale pre-training endows our method with powerful transferability across a wide range of VLMs. Extensive experiments on five mainstream open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) across three multimodal tasks (image-text retrieval, multimodal classification, and image captioning) demonstrate the effectiveness of our attack. Additionally, we successfully transfer AnyAttack to multiple commercial VLMs, including Google's Gemini, Claude's Sonnet, and Microsoft's Copilot. These results reveal an unprecedented risk to VLMs, highlighting the need for effective countermeasures.



## **35. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

cs.CR

10 pages, 5 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.02298v2) [paper-pdf](http://arxiv.org/pdf/2410.02298v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.



## **36. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

cs.AI

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2406.12257v2) [paper-pdf](http://arxiv.org/pdf/2406.12257v2)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CLEANGEN, to mitigate backdoor attacks for generation tasks in LLMs. CLEANGEN is a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CLEANGEN is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CLEANGEN to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CLEANGEN against five SOTA backdoor attacks. Our results show that CLEANGEN achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CLEANGEN maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.



## **37. Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm**

cs.CL

Under Review

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2409.14119v3) [paper-pdf](http://arxiv.org/pdf/2409.14119v3)

**Authors**: Jaehan Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin

**Abstract**: Parameter-efficient fine-tuning (PEFT) has become a key training strategy for large language models. However, its reliance on fewer trainable parameters poses security risks, such as task-agnostic backdoors. Despite their severe impact on a wide range of tasks, there is no practical defense solution available that effectively counters task-agnostic backdoors within the context of PEFT. In this study, we introduce Obliviate, a PEFT-integrable backdoor defense. We develop two techniques aimed at amplifying benign neurons within PEFT layers and penalizing the influence of trigger tokens. Our evaluations across three major PEFT architectures show that our method can significantly reduce the attack success rate of the state-of-the-art task-agnostic backdoors (83.6%$\downarrow$). Furthermore, our method exhibits robust defense capabilities against both task-specific backdoors and adaptive attacks. Source code will be obtained at https://github.com/obliviateARR/Obliviate.



## **38. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

cs.CR

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2404.18816v2) [paper-pdf](http://arxiv.org/pdf/2404.18816v2)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous learning-based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing learning-based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Then, using our carefully crafted multi-view prompt templates, it guides the LLM to generate function descriptions and behavioral summaries for each view, enabling deep semantic analysis of the views. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the human-readable diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline methods. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.



## **39. Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks**

cs.LG

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04234v1) [paper-pdf](http://arxiv.org/pdf/2410.04234v1)

**Authors**: Zi Wang, Divyam Anshumaan, Ashish Hooda, Yudong Chen, Somesh Jha

**Abstract**: Optimization methods are widely employed in deep learning to identify and mitigate undesired model responses. While gradient-based techniques have proven effective for image models, their application to language models is hindered by the discrete nature of the input space. This study introduces a novel optimization approach, termed the \emph{functional homotopy} method, which leverages the functional duality between model training and input generation. By constructing a series of easy-to-hard optimization problems, we iteratively solve these problems using principles derived from established homotopy methods. We apply this approach to jailbreak attack synthesis for large language models (LLMs), achieving a $20\%-30\%$ improvement in success rate over existing methods in circumventing established safe open-source models such as Llama-2 and Llama-3.



## **40. Adversarial Suffixes May Be Features Too!**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.00451v2) [paper-pdf](http://arxiv.org/pdf/2410.00451v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including those triggered by adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets, i.e., even in the absence of harmful content. This highlights the critical risk posed by dominating benign features in the training data and calls for further research to reinforce LLM safety alignment. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.



## **41. Automated Progressive Red Teaming**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2407.03876v2) [paper-pdf](http://arxiv.org/pdf/2407.03876v2)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).



## **42. Harnessing Task Overload for Scalable Jailbreak Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04190v1) [paper-pdf](http://arxiv.org/pdf/2410.04190v1)

**Authors**: Yiting Dong, Guobin Shen, Dongcheng Zhao, Xiang He, Yi Zeng

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attacks that bypass their safety mechanisms. Existing attack methods are fixed or specifically tailored for certain models and cannot flexibly adjust attack strength, which is critical for generalization when attacking models of various sizes. We introduce a novel scalable jailbreak attack that preempts the activation of an LLM's safety policies by occupying its computational resources. Our method involves engaging the LLM in a resource-intensive preliminary task - a Character Map lookup and decoding process - before presenting the target instruction. By saturating the model's processing capacity, we prevent the activation of safety protocols when processing the subsequent instruction. Extensive experiments on state-of-the-art LLMs demonstrate that our method achieves a high success rate in bypassing safety measures without requiring gradient access, manual prompt engineering. We verified our approach offers a scalable attack that quantifies attack strength and adapts to different model scales at the optimal strength. We shows safety policies of LLMs might be more susceptible to resource constraints. Our findings reveal a critical vulnerability in current LLM safety designs, highlighting the need for more robust defense strategies that account for resource-intense condition.



## **43. Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems**

cs.CR

31 pages, including main paper, references, and appendix

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2405.20774v2) [paper-pdf](http://arxiv.org/pdf/2405.20774v2)

**Authors**: Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, Qi Zhu

**Abstract**: Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.



## **44. ASPIRER: Bypassing System Prompts With Permutation-based Backdoors in LLMs**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04009v1) [paper-pdf](http://arxiv.org/pdf/2410.04009v1)

**Authors**: Lu Yan, Siyuan Cheng, Xuan Chen, Kaiyuan Zhang, Guangyu Shen, Zhuo Zhang, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) have become integral to many applications, with system prompts serving as a key mechanism to regulate model behavior and ensure ethical outputs. In this paper, we introduce a novel backdoor attack that systematically bypasses these system prompts, posing significant risks to the AI supply chain. Under normal conditions, the model adheres strictly to its system prompts. However, our backdoor allows malicious actors to circumvent these safeguards when triggered. Specifically, we explore a scenario where an LLM provider embeds a covert trigger within the base model. A downstream deployer, unaware of the hidden trigger, fine-tunes the model and offers it as a service to users. Malicious actors can purchase the trigger from the provider and use it to exploit the deployed model, disabling system prompts and achieving restricted outcomes. Our attack utilizes a permutation trigger, which activates only when its components are arranged in a precise order, making it computationally challenging to detect or reverse-engineer. We evaluate our approach on five state-of-the-art models, demonstrating that our method achieves an attack success rate (ASR) of up to 99.50% while maintaining a clean accuracy (CACC) of 98.58%, even after defensive fine-tuning. These findings highlight critical vulnerabilities in LLM deployment pipelines and underscore the need for stronger defenses.



## **45. Detecting Machine-Generated Long-Form Content with Latent-Space Variables**

cs.CL

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03856v1) [paper-pdf](http://arxiv.org/pdf/2410.03856v1)

**Authors**: Yufei Tian, Zeyu Pan, Nanyun Peng

**Abstract**: The increasing capability of large language models (LLMs) to generate fluent long-form texts is presenting new challenges in distinguishing machine-generated outputs from human-written ones, which is crucial for ensuring authenticity and trustworthiness of expressions. Existing zero-shot detectors primarily focus on token-level distributions, which are vulnerable to real-world domain shifts, including different prompting and decoding strategies, and adversarial attacks. We propose a more robust method that incorporates abstract elements, such as event transitions, as key deciding factors to detect machine versus human texts by training a latent-space model on sequences of events or topics derived from human-written texts. In three different domains, machine-generated texts, which are originally inseparable from human texts on the token level, can be better distinguished with our latent-space model, leading to a 31% improvement over strong baselines such as DetectGPT. Our analysis further reveals that, unlike humans, modern LLMs like GPT-4 generate event triggers and their transitions differently, an inherent disparity that helps our method to robustly detect machine-generated texts.



## **46. Developing Assurance Cases for Adversarial Robustness and Regulatory Compliance in LLMs**

cs.CR

Accepted to the ASSURE 2024 workshop

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.05304v1) [paper-pdf](http://arxiv.org/pdf/2410.05304v1)

**Authors**: Tomas Bueno Momcilovic, Dian Balta, Beat Buesser, Giulio Zizzo, Mark Purcell

**Abstract**: This paper presents an approach to developing assurance cases for adversarial robustness and regulatory compliance in large language models (LLMs). Focusing on both natural and code language tasks, we explore the vulnerabilities these models face, including adversarial attacks based on jailbreaking, heuristics, and randomization. We propose a layered framework incorporating guardrails at various stages of LLM deployment, aimed at mitigating these attacks and ensuring compliance with the EU AI Act. Our approach includes a meta-layer for dynamic risk management and reasoning, crucial for addressing the evolving nature of LLM vulnerabilities. We illustrate our method with two exemplary assurance cases, highlighting how different contexts demand tailored strategies to ensure robust and compliant AI systems.



## **47. RAFT: Realistic Attacks to Fool Text Detectors**

cs.CL

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03658v1) [paper-pdf](http://arxiv.org/pdf/2410.03658v1)

**Authors**: James Wang, Ran Li, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) have exhibited remarkable fluency across various tasks. However, their unethical applications, such as disseminating disinformation, have become a growing concern. Although recent works have proposed a number of LLM detection methods, their robustness and reliability remain unclear. In this paper, we present RAFT: a grammar error-free black-box attack against existing LLM detectors. In contrast to previous attacks for language models, our method exploits the transferability of LLM embeddings at the word-level while preserving the original text quality. We leverage an auxiliary embedding to greedily select candidate words to perturb against the target detector. Experiments reveal that our attack effectively compromises all detectors in the study across various domains by up to 99%, and are transferable across source models. Manual human evaluation studies show our attacks are realistic and indistinguishable from original human-written text. We also show that examples generated by RAFT can be used to train adversarially robust detectors. Our work shows that current LLM detectors are not adversarially robust, underscoring the urgent need for more resilient detection mechanisms.



## **48. Buckle Up: Robustifying LLMs at Every Customization Stage via Data Curation**

cs.CR

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.02220v2) [paper-pdf](http://arxiv.org/pdf/2410.02220v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Luoxi Tang, Chenyu You, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are extensively adapted for downstream applications through a process known as "customization," with fine-tuning being a common method for integrating domain-specific expertise. However, recent studies have revealed a vulnerability that tuning LLMs with malicious samples can compromise their robustness and amplify harmful content, an attack known as "jailbreaking." To mitigate such attack, we propose an effective defensive framework utilizing data curation to revise commonsense texts and enhance their safety implication from the perspective of LLMs. The curated texts can mitigate jailbreaking attacks at every stage of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize jailbreaking risks, or after customization to restore the compromised models. Since the curated data strengthens LLMs through the standard fine-tuning workflow, we do not introduce additional modules during LLM inference, thereby preserving the original customization process. Experimental results demonstrate a substantial reduction in jailbreaking effects, with up to a 100% success in generating responsible responses. Notably, our method is effective even with commonsense texts, which are often more readily available than safety-relevant data. With the every-stage defensive framework and supporting experimental performance, this work represents a significant advancement in mitigating jailbreaking risks and ensuring the secure customization of LLMs.



## **49. Jailbreaking as a Reward Misspecification Problem**

cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.14393v3) [paper-pdf](http://arxiv.org/pdf/2406.14393v3)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.



## **50. AutoPenBench: Benchmarking Generative Agents for Penetration Testing**

cs.CR

Codes for the benchmark:  https://github.com/lucagioacchini/auto-pen-bench Codes for the paper  experiments: https://github.com/lucagioacchini/genai-pentest-paper

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03225v1) [paper-pdf](http://arxiv.org/pdf/2410.03225v1)

**Authors**: Luca Gioacchini, Marco Mellia, Idilio Drago, Alexander Delsanto, Giuseppe Siracusano, Roberto Bifulco

**Abstract**: Generative AI agents, software systems powered by Large Language Models (LLMs), are emerging as a promising approach to automate cybersecurity tasks. Among the others, penetration testing is a challenging field due to the task complexity and the diverse strategies to simulate cyber-attacks. Despite growing interest and initial studies in automating penetration testing with generative agents, there remains a significant gap in the form of a comprehensive and standard framework for their evaluation and development. This paper introduces AutoPenBench, an open benchmark for evaluating generative agents in automated penetration testing. We present a comprehensive framework that includes 33 tasks, each representing a vulnerable system that the agent has to attack. Tasks are of increasing difficulty levels, including in-vitro and real-world scenarios. We assess the agent performance with generic and specific milestones that allow us to compare results in a standardised manner and understand the limits of the agent under test. We show the benefits of AutoPenBench by testing two agent architectures: a fully autonomous and a semi-autonomous supporting human interaction. We compare their performance and limitations. For example, the fully autonomous agent performs unsatisfactorily achieving a 21% Success Rate (SR) across the benchmark, solving 27% of the simple tasks and only one real-world task. In contrast, the assisted agent demonstrates substantial improvements, with 64% of SR. AutoPenBench allows us also to observe how different LLMs like GPT-4o or OpenAI o1 impact the ability of the agents to complete the tasks. We believe that our benchmark fills the gap with a standard and flexible framework to compare penetration testing agents on a common ground. We hope to extend AutoPenBench along with the research community by making it available under https://github.com/lucagioacchini/auto-pen-bench.



