# Latest Large Language Model Attack Papers
**update at 2024-02-21 10:58:20**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Prompt Stealing Attacks Against Large Language Models**

cs.CR

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12959v1) [paper-pdf](http://arxiv.org/pdf/2402.12959v1)

**Authors**: Zeyang Sha, Yang Zhang

**Abstract**: The increasing reliance on large language models (LLMs) such as ChatGPT in various fields emphasizes the importance of ``prompt engineering,'' a technology to improve the quality of model outputs. With companies investing significantly in expert prompt engineers and educational resources rising to meet market demand, designing high-quality prompts has become an intriguing challenge. In this paper, we propose a novel attack against LLMs, named prompt stealing attacks. Our proposed prompt stealing attack aims to steal these well-designed prompts based on the generated answers. The prompt stealing attack contains two primary modules: the parameter extractor and the prompt reconstruction. The goal of the parameter extractor is to figure out the properties of the original prompts. We first observe that most prompts fall into one of three categories: direct prompt, role-based prompt, and in-context prompt. Our parameter extractor first tries to distinguish the type of prompts based on the generated answers. Then, it can further predict which role or how many contexts are used based on the types of prompts. Following the parameter extractor, the prompt reconstructor can be used to reconstruct the original prompts based on the generated answers and the extracted features. The final goal of the prompt reconstructor is to generate the reversed prompts, which are similar to the original prompts. Our experimental results show the remarkable performance of our proposed attacks. Our proposed attacks add a new dimension to the study of prompt engineering and call for more attention to the security issues on LLMs.



## **2. Measuring Impacts of Poisoning on Model Parameters and Neuron Activations: A Case Study of Poisoning CodeBERT**

cs.SE

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12936v1) [paper-pdf](http://arxiv.org/pdf/2402.12936v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Navid Ayoobi, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have revolutionized software development practices, yet concerns about their safety have arisen, particularly regarding hidden backdoors, aka trojans. Backdoor attacks involve the insertion of triggers into training data, allowing attackers to manipulate the behavior of the model maliciously. In this paper, we focus on analyzing the model parameters to detect potential backdoor signals in code models. Specifically, we examine attention weights and biases, activation values, and context embeddings of the clean and poisoned CodeBERT models. Our results suggest noticeable patterns in activation values and context embeddings of poisoned samples for the poisoned CodeBERT model; however, attention weights and biases do not show any significant differences. This work contributes to ongoing efforts in white-box detection of backdoor signals in LLMs of code through the analysis of parameters and activations.



## **3. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12343v1) [paper-pdf](http://arxiv.org/pdf/2402.12343v1)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) need to undergo safety alignment to ensure safe conversations with humans. However, in this work, we introduce an inference-time attack framework, demonstrating that safety alignment can also unintentionally facilitate harmful outcomes under adversarial manipulation. This framework, named Emulated Disalignment (ED), adversely combines a pair of open-source pre-trained and safety-aligned language models in the output space to produce a harmful language model without any training. Our experiments with ED across three datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rate in 43 out of 48 evaluation subsets by a large margin. Crucially, our findings highlight the importance of reevaluating the practice of open-sourcing language models even after safety alignment.



## **4. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12336v1) [paper-pdf](http://arxiv.org/pdf/2402.12336v1)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many vision-language models (VLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (VLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of VLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the VLM is required. The code and robust models are available at https://github.com/chs20/RobustVLM



## **5. Polarization of Autonomous Generative AI Agents Under Echo Chambers**

cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12212v1) [paper-pdf](http://arxiv.org/pdf/2402.12212v1)

**Authors**: Masaya Ohagi

**Abstract**: Online social networks often create echo chambers where people only hear opinions reinforcing their beliefs. An echo chamber often generates polarization, leading to conflicts caused by people with radical opinions, such as the January 6, 2021, attack on the US Capitol. The echo chamber has been viewed as a human-specific problem, but this implicit assumption is becoming less reasonable as large language models, such as ChatGPT, acquire social abilities. In response to this situation, we investigated the potential for polarization to occur among a group of autonomous AI agents based on generative language models in an echo chamber environment. We had AI agents discuss specific topics and analyzed how the group's opinions changed as the discussion progressed. As a result, we found that the group of agents based on ChatGPT tended to become polarized in echo chamber environments. The analysis of opinion transitions shows that this result is caused by ChatGPT's high prompt understanding ability to update its opinion by considering its own and surrounding agents' opinions. We conducted additional experiments to investigate under what specific conditions AI agents tended to polarize. As a result, we identified factors that strongly influence polarization, such as the agent's persona. These factors should be monitored to prevent the polarization of AI agents.



## **6. On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities**

cs.RO

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.10340v2) [paper-pdf](http://arxiv.org/pdf/2402.10340v2)

**Authors**: Xiyang Wu, Ruiqi Xian, Tianrui Guan, Jing Liang, Souradip Chakraborty, Fuxiao Liu, Brian Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works have focused on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation, navigation, etc. However, such integration can introduce significant vulnerabilities, in terms of their susceptibility to adversarial attacks due to the language models, potentially leading to catastrophic consequences. By examining recent works at the interface of LLMs/VLMs and robotics, we show that it is easy to manipulate or misguide the robot's actions, leading to safety hazards. We define and provide examples of several plausible adversarial attacks, and conduct experiments on three prominent robot frameworks integrated with a language model, including KnowNo VIMA, and Instruct2Act, to assess their susceptibility to these attacks. Our empirical findings reveal a striking vulnerability of LLM/VLM-robot integrated systems: simple adversarial attacks can significantly undermine the effectiveness of LLM/VLM-robot integrated systems. Specifically, our data demonstrate an average performance deterioration of 21.2% under prompt attacks and a more alarming 30.2% under perception attacks. These results underscore the critical need for robust countermeasures to ensure the safe and reliable deployment of the advanced LLM/VLM-based robotic systems.



## **7. SPML: A DSL for Defending Language Models Against Prompt Attacks**

cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.11755v1) [paper-pdf](http://arxiv.org/pdf/2402.11755v1)

**Authors**: Reshabh K Sharma, Vinayak Gupta, Dan Grossman

**Abstract**: Large language models (LLMs) have profoundly transformed natural language applications, with a growing reliance on instruction-based definitions for designing chatbots. However, post-deployment the chatbot definitions are fixed and are vulnerable to attacks by malicious users, emphasizing the need to prevent unethical applications and financial losses. Existing studies explore user prompts' impact on LLM-based chatbots, yet practical methods to contain attacks on application-specific chatbots remain unexplored. This paper presents System Prompt Meta Language (SPML), a domain-specific language for refining prompts and monitoring the inputs to the LLM-based chatbots. SPML actively checks attack prompts, ensuring user inputs align with chatbot definitions to prevent malicious execution on the LLM backbone, optimizing costs. It also streamlines chatbot definition crafting with programming language capabilities, overcoming natural language design challenges. Additionally, we introduce a groundbreaking benchmark with 1.8k system prompts and 20k user inputs, offering the inaugural language and benchmark for chatbot definition evaluation. Experiments across datasets demonstrate SPML's proficiency in understanding attacker prompts, surpassing models like GPT-4, GPT-3.5, and LLAMA. Our data and codes are publicly available at: https://prompt-compiler.github.io/SPML/.



## **8. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.11753v1) [paper-pdf](http://arxiv.org/pdf/2402.11753v1)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs.



## **9. Vaccine: Perturbation-aware Alignment for Large Language Model**

cs.LG

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.01109v2) [paper-pdf](http://arxiv.org/pdf/2402.01109v2)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.



## **10. Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks**

cs.CL

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11638v1) [paper-pdf](http://arxiv.org/pdf/2402.11638v1)

**Authors**: Yichen Wang, Shangbin Feng, Abe Bohan Hou, Xiao Pu, Chao Shen, Xiaoming Liu, Yulia Tsvetkov, Tianxing He

**Abstract**: The widespread use of large language models (LLMs) is increasing the demand for methods that detect machine-generated text to prevent misuse. The goal of our study is to stress test the detectors' robustness to malicious attacks under realistic scenarios. We comprehensively study the robustness of popular machine-generated text detectors under attacks from diverse categories: editing, paraphrasing, prompting, and co-generating. Our attacks assume limited access to the generator LLMs, and we compare the performance of detectors on different attacks under different budget levels. Our experiments reveal that almost none of the existing detectors remain robust under all the attacks, and all detectors exhibit different loopholes. Averaging all detectors, the performance drops by 35% across all attacks. Further, we investigate the reasons behind these defects and propose initial out-of-the-box patches to improve robustness.



## **11. OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples**

cs.CL

AAAI 2024 camera ready. Code and dataset available at  https://github.com/ryuryukke/OUTFOX

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2307.11729v3) [paper-pdf](http://arxiv.org/pdf/2307.11729v3)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.



## **12. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

cs.CL

Pre-print, code is available at https://github.com/NJUNLP/ReNeLLM

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2311.08268v2) [paper-pdf](http://arxiv.org/pdf/2311.08268v2)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, compromising generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.



## **13. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

cs.CL

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2311.11509v3) [paper-pdf](http://arxiv.org/pdf/2311.11509v3)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that mislead LLMs into generating incorrect or undesired outputs. Previous work has revealed that with relatively simple yet effective attacks based on discrete optimization, it is possible to generate adversarial prompts that bypass moderation and alignment of the models. This vulnerability to adversarial prompts underscores a significant concern regarding the robustness and reliability of LLMs. Our work aims to address this concern by introducing a novel approach to detecting adversarial prompts at a token level, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity, where tokens predicted with high probability are considered normal, and those exhibiting high perplexity are flagged as adversarial. Additionaly, our method also integrates context understanding by incorporating neighboring token information to encourage the detection of contiguous adversarial prompt sequences. To this end, we design two algorithms for adversarial prompt detection: one based on optimization techniques and another on Probabilistic Graphical Models (PGM). Both methods are equipped with efficient solving methods, ensuring efficient adversarial prompt detection. Our token-level detection result can be visualized as heatmap overlays on the text sequence, allowing for a clearer and more intuitive representation of which part of the text may contain adversarial prompts.



## **14. Effective Prompt Extraction from Language Models**

cs.CL

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2307.06865v2) [paper-pdf](http://arxiv.org/pdf/2307.06865v2)

**Authors**: Yiming Zhang, Nicholas Carlini, Daphne Ippolito

**Abstract**: The text generated by large language models is commonly controlled by prompting, where a prompt prepended to a user's query guides the model's output. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold. However, anecdotal reports have shown adversarial users employing prompt extraction attacks to recover these prompts. In this paper, we present a framework for systematically measuring the effectiveness of these attacks. In experiments with 3 different sources of prompts and 11 underlying large language models, we find that simple text-based attacks can in fact reveal prompts with high probability. Our framework determines with high precision whether an extracted prompt is the actual secret prompt, rather than a model hallucination. Prompt extraction experiments on real systems such as Bing Chat and ChatGPT suggest that system prompts can be revealed by an adversary despite existing defenses in place.



## **15. Can Large Language Models perform Relation-based Argument Mining?**

cs.CL

10 pages, 9 figures, submitted to ACL 2024

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2402.11243v1) [paper-pdf](http://arxiv.org/pdf/2402.11243v1)

**Authors**: Deniz Gorur, Antonio Rago, Francesca Toni

**Abstract**: Argument mining (AM) is the process of automatically extracting arguments, their components and/or relations amongst arguments and components from text. As the number of platforms supporting online debate increases, the need for AM becomes ever more urgent, especially in support of downstream tasks. Relation-based AM (RbAM) is a form of AM focusing on identifying agreement (support) and disagreement (attack) relations amongst arguments. RbAM is a challenging classification task, with existing methods failing to perform satisfactorily. In this paper, we show that general-purpose Large Language Models (LLMs), appropriately primed and prompted, can significantly outperform the best performing (RoBERTa-based) baseline. Specifically, we experiment with two open-source LLMs (Llama-2 and Mistral) with ten datasets.



## **16. Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents**

cs.CR

The first two authors contribute equally. Code and data are available  at https://github.com/lancopku/agent-backdoor-attacks

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2402.11208v1) [paper-pdf](http://arxiv.org/pdf/2402.11208v1)

**Authors**: Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, Xu Sun

**Abstract**: Leveraging the rapid development of Large Language Models LLMs, LLM-based agents have been developed to handle various real-world applications, including finance, healthcare, and shopping, etc. It is crucial to ensure the reliability and security of LLM-based agents during applications. However, the safety issues of LLM-based agents are currently under-explored. In this work, we take the first step to investigate one of the typical safety threats, backdoor attack, to LLM-based agents. We first formulate a general framework of agent backdoor attacks, then we present a thorough analysis on the different forms of agent backdoor attacks. Specifically, from the perspective of the final attacking outcomes, the attacker can either choose to manipulate the final output distribution, or only introduce malicious behavior in the intermediate reasoning process, while keeping the final output correct. Furthermore, the former category can be divided into two subcategories based on trigger locations: the backdoor trigger can be hidden either in the user query or in an intermediate observation returned by the external environment. We propose the corresponding data poisoning mechanisms to implement the above variations of agent backdoor attacks on two typical agent tasks, web shopping and tool utilization. Extensive experiments show that LLM-based agents suffer severely from backdoor attacks, indicating an urgent need for further research on the development of defenses against backdoor attacks on LLM-based agents. Warning: This paper may contain biased content.



## **17. Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering**

cs.CL

13 pages, 9 figures

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2401.06824v2) [paper-pdf](http://arxiv.org/pdf/2401.06824v2)

**Authors**: Tianlong Li, Shihan Dou, Wenhao Liu, Muling Wu, Changze Lv, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: Jailbreaking techniques aim to probe the boundaries of safety in large language models (LLMs) by inducing them to generate toxic responses to malicious queries, a significant concern within the LLM community. While existing jailbreaking methods primarily rely on prompt engineering, altering inputs to evade LLM safety mechanisms, they suffer from low attack success rates and significant time overheads, rendering them inflexible. To overcome these limitations, we propose a novel jailbreaking approach, named Jailbreaking LLMs through Representation Engineering (JRE). Our method requires only a small number of query pairs to extract ``safety patterns'' that can be used to circumvent the target model's defenses, achieving unprecedented jailbreaking performance. Building upon these findings, we also introduce a novel defense framework inspired by JRE principles, which demonstrates notable effectiveness. Extensive experimentation confirms the superior performance of the JRE attacks and the robustness of the JRE defense framework. We hope this study contributes to advancing the understanding of model safety issues through the lens of representation engineering.



## **18. VQAttack: Transferable Adversarial Attacks on Visual Question Answering via Pre-trained Models**

cs.CV

AAAI 2024, 11 pages

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.11083v1) [paper-pdf](http://arxiv.org/pdf/2402.11083v1)

**Authors**: Ziyi Yin, Muchao Ye, Tianrong Zhang, Jiaqi Wang, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**Abstract**: Visual Question Answering (VQA) is a fundamental task in computer vision and natural language process fields. Although the ``pre-training & finetuning'' learning paradigm significantly improves the VQA performance, the adversarial robustness of such a learning paradigm has not been explored. In this paper, we delve into a new problem: using a pre-trained multimodal source model to create adversarial image-text pairs and then transferring them to attack the target VQA models. Correspondingly, we propose a novel VQAttack model, which can iteratively generate both image and text perturbations with the designed modules: the large language model (LLM)-enhanced image attack and the cross-modal joint attack module. At each iteration, the LLM-enhanced image attack module first optimizes the latent representation-based loss to generate feature-level image perturbations. Then it incorporates an LLM to further enhance the image perturbations by optimizing the designed masked answer anti-recovery loss. The cross-modal joint attack module will be triggered at a specific iteration, which updates the image and text perturbations sequentially. Notably, the text perturbation updates are based on both the learned gradients in the word embedding space and word synonym-based substitution. Experimental results on two VQA datasets with five validated models demonstrate the effectiveness of the proposed VQAttack in the transferable attack setting, compared with state-of-the-art baselines. This work reveals a significant blind spot in the ``pre-training & fine-tuning'' paradigm on VQA tasks. Source codes will be released.



## **19. ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages**

cs.CL

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10753v1) [paper-pdf](http://arxiv.org/pdf/2402.10753v1)

**Authors**: Junjie Ye, Sixian Li, Guanyu Li, Caishuang Huang, Songyang Gao, Yilong Wu, Qi Zhang, Tao Gui, Xuanjing Huang

**Abstract**: Tool learning is widely acknowledged as a foundational approach or deploying large language models (LLMs) in real-world scenarios. While current research primarily emphasizes leveraging tools to augment LLMs, it frequently neglects emerging safety considerations tied to their application. To fill this gap, we present $ToolSword$, a comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning. Specifically, ToolSword delineates six safety scenarios for LLMs in tool learning, encompassing $malicious$ $queries$ and $jailbreak$ $attacks$ in the input stage, $noisy$ $misdirection$ and $risky$ $cues$ in the execution stage, and $harmful$ $feedback$ and $error$ $conflicts$ in the output stage. Experiments conducted on 11 open-source and closed-source LLMs reveal enduring safety challenges in tool learning, such as handling harmful queries, employing risky tools, and delivering detrimental feedback, which even GPT-4 is susceptible to. Moreover, we conduct further studies with the aim of fostering research on tool learning safety. The data is released in https://github.com/Junjie-Ye/ToolSword.



## **20. Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**

cs.CV

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.00626v2) [paper-pdf](http://arxiv.org/pdf/2402.00626v2)

**Authors**: Maan Qraitem, Nazia Tasnim, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Typographic Attacks, which involve pasting misleading text onto an image, were noted to harm the performance of Vision-Language Models like CLIP. However, the susceptibility of recent Large Vision-Language Models to these attacks remains understudied. Furthermore, prior work's Typographic attacks against CLIP randomly sample a misleading class from a predefined set of categories. However, this simple strategy misses more effective attacks that exploit LVLM(s) stronger language skills. To address these issues, we first introduce a benchmark for testing Typographic attacks against LVLM(s). Moreover, we introduce two novel and more effective \textit{Self-Generated} attacks which prompt the LVLM to generate an attack against itself: 1) Class Based Attack where the LVLM (e.g. LLaVA) is asked which deceiving class is most similar to the target class and 2) Descriptive Attacks where a more advanced LVLM (e.g. GPT4-V) is asked to recommend a Typographic attack that includes both a deceiving class and description. Using our benchmark, we uncover that Self-Generated attacks pose a significant threat, reducing LVLM(s) classification performance by up to 33\%. We also uncover that attacks generated by one model (e.g. GPT-4V or LLaVA) are effective against the model itself and other models like InstructBLIP and MiniGPT4. Code: \url{https://github.com/mqraitem/Self-Gen-Typo-Attack}



## **21. Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning**

cs.CL

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2401.05949v4) [paper-pdf](http://arxiv.org/pdf/2401.05949v4)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we design a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning demonstration prompts, which can make models behave in alignment with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 180B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models.



## **22. Humans or LLMs as the Judge? A Study on Judgement Biases**

cs.CL

19 pages

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.10669v2) [paper-pdf](http://arxiv.org/pdf/2402.10669v2)

**Authors**: Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, Benyou Wang

**Abstract**: Adopting human and large language models (LLM) as judges (\textit{a.k.a} human- and LLM-as-a-judge) for evaluating the performance of existing LLMs has recently gained attention. Nonetheless, this approach concurrently introduces potential biases from human and LLM judges, questioning the reliability of the evaluation results. In this paper, we propose a novel framework for investigating 5 types of biases for LLM and human judges. We curate a dataset with 142 samples referring to the revised Bloom's Taxonomy and conduct thousands of human and LLM evaluations. Results show that human and LLM judges are vulnerable to perturbations to various degrees, and that even the most cutting-edge judges possess considerable biases. We further exploit their weakness and conduct attacks on LLM judges. We hope that our work can notify the community of the vulnerability of human- and LLM-as-a-judge against perturbations, as well as the urgency of developing robust evaluation systems.



## **23. Jailbreaking Proprietary Large Language Models using Word Substitution Cipher**

cs.CL

15 pages

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10601v1) [paper-pdf](http://arxiv.org/pdf/2402.10601v1)

**Authors**: Divij Handa, Advait Chirmule, Bimal Gajera, Chitta Baral

**Abstract**: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbreaking approach on state-of-the-art proprietary models including ChatGPT, GPT-4, and Gemini-Pro. Additionally, we discuss the over-defensiveness of these models. We believe that our work will encourage further research in making these LLMs more robust while maintaining their decoding capabilities.



## **24. Text Embedding Inversion Security for Multilingual Language Models**

cs.CL

18 pages, 17 Tables, 6 Figures

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2401.12192v2) [paper-pdf](http://arxiv.org/pdf/2401.12192v2)

**Authors**: Yiyi Chen, Heather Lent, Johannes Bjerva

**Abstract**: Textual data is often represented as realnumbered embeddings in NLP, particularly with the popularity of large language models (LLMs) and Embeddings as a Service (EaaS). However, storing sensitive information as embeddings can be vulnerable to security breaches, as research shows that text can be reconstructed from embeddings, even without knowledge of the underlying model. While defence mechanisms have been explored, these are exclusively focused on English, leaving other languages vulnerable to attacks. This work explores LLM security through multilingual embedding inversion. We define the problem of black-box multilingual and cross-lingual inversion attacks, and thoroughly explore their potential implications. Our findings suggest that multilingual LLMs may be more vulnerable to inversion attacks, in part because English based defences may be ineffective. To alleviate this, we propose a simple masking defense effective for both monolingual and multilingual models. This study is the first to investigate multilingual inversion attacks, shedding light on the differences in attacks and defenses across monolingual and multilingual settings.



## **25. Zero-shot sampling of adversarial entities in biomedical question answering**

cs.CL

20 pages incl. appendix, under review

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10527v1) [paper-pdf](http://arxiv.org/pdf/2402.10527v1)

**Authors**: R. Patrick Xian, Alex J. Lee, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks successfully manipulate token-wise Shapley value explanations, which become deceptive in the adversarial setting. Our investigations illustrate the brittleness of domain knowledge in LLMs and reveal a shortcoming of standard evaluations for high-capacity models.



## **26. A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents**

cs.CL

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10196v1) [paper-pdf](http://arxiv.org/pdf/2402.10196v1)

**Authors**: Lingbo Mo, Zeyi Liao, Boyuan Zheng, Yu Su, Chaowei Xiao, Huan Sun

**Abstract**: Language agents powered by large language models (LLMs) have seen exploding development. Their capability of using language as a vehicle for thought and communication lends an incredible level of flexibility and versatility. People have quickly capitalized on this capability to connect LLMs to a wide range of external components and environments: databases, tools, the Internet, robotic embodiment, etc. Many believe an unprecedentedly powerful automation technology is emerging. However, new automation technologies come with new safety risks, especially for intricate systems like language agents. There is a surprisingly large gap between the speed and scale of their development and deployment and our understanding of their safety risks. Are we building a house of cards? In this position paper, we present the first systematic effort in mapping adversarial attacks against language agents. We first present a unified conceptual framework for agents with three major components: Perception, Brain, and Action. Under this framework, we present a comprehensive discussion and propose 12 potential attack scenarios against different components of an agent, covering different attack strategies (e.g., input manipulation, adversarial demonstrations, jailbreaking, backdoors). We also draw connections to successful attack strategies previously applied to LLMs. We emphasize the urgency to gain a thorough understanding of language agent risks before their widespread deployment.



## **27. Exploring the Adversarial Capabilities of Large Language Models**

cs.AI

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09132v2) [paper-pdf](http://arxiv.org/pdf/2402.09132v2)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.



## **28. Rapid Adoption, Hidden Risks: The Dual Impact of Large Language Model Customization**

cs.CR

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09179v2) [paper-pdf](http://arxiv.org/pdf/2402.09179v2)

**Authors**: Rui Zhang, Hongwei Li, Rui Wen, Wenbo Jiang, Yuan Zhang, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The increasing demand for customized Large Language Models (LLMs) has led to the development of solutions like GPTs. These solutions facilitate tailored LLM creation via natural language prompts without coding. However, the trustworthiness of third-party custom versions of LLMs remains an essential concern. In this paper, we propose the first instruction backdoor attacks against applications integrated with untrusted customized LLMs (e.g., GPTs). Specifically, these attacks embed the backdoor into the custom version of LLMs by designing prompts with backdoor instructions, outputting the attacker's desired result when inputs contain the pre-defined triggers. Our attack includes 3 levels of attacks: word-level, syntax-level, and semantic-level, which adopt different types of triggers with progressive stealthiness. We stress that our attacks do not require fine-tuning or any modification to the backend LLMs, adhering strictly to GPTs development guidelines. We conduct extensive experiments on 4 prominent LLMs and 5 benchmark text classification datasets. The results show that our instruction backdoor attacks achieve the desired attack performance without compromising utility. Additionally, we propose an instruction-ignoring defense mechanism and demonstrate its partial effectiveness in mitigating such attacks. Our findings highlight the vulnerability and the potential risks of LLM customization such as GPTs.



## **29. AbuseGPT: Abuse of Generative AI ChatBots to Create Smishing Campaigns**

cs.CR

6 pages, 12 figures, published in ISDFS 2024

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09728v1) [paper-pdf](http://arxiv.org/pdf/2402.09728v1)

**Authors**: Ashfak Md Shibli, Mir Mehedi A. Pritom, Maanak Gupta

**Abstract**: SMS phishing, also known as "smishing", is a growing threat that tricks users into disclosing private information or clicking into URLs with malicious content through fraudulent mobile text messages. In recent past, we have also observed a rapid advancement of conversational generative AI chatbot services (e.g., OpenAI's ChatGPT, Google's BARD), which are powered by pre-trained large language models (LLMs). These AI chatbots certainly have a lot of utilities but it is not systematically understood how they can play a role in creating threats and attacks. In this paper, we propose AbuseGPT method to show how the existing generative AI-based chatbot services can be exploited by attackers in real world to create smishing texts and eventually lead to craftier smishing campaigns. To the best of our knowledge, there is no pre-existing work that evidently shows the impacts of these generative text-based models on creating SMS phishing. Thus, we believe this study is the first of its kind to shed light on this emerging cybersecurity threat. We have found strong empirical evidences to show that attackers can exploit ethical standards in the existing generative AI-based chatbot services by crafting prompt injection attacks to create newer smishing campaigns. We also discuss some future research directions and guidelines to protect the abuse of generative AI-based services and safeguard users from smishing attacks.



## **30. Detecting Phishing Sites Using ChatGPT**

cs.CR

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2306.05816v2) [paper-pdf](http://arxiv.org/pdf/2306.05816v2)

**Authors**: Takashi Koide, Naoki Fukushi, Hiroki Nakano, Daiki Chiba

**Abstract**: The emergence of Large Language Models (LLMs), including ChatGPT, is having a significant impact on a wide range of fields. While LLMs have been extensively researched for tasks such as code generation and text synthesis, their application in detecting malicious web content, particularly phishing sites, has been largely unexplored. To combat the rising tide of cyber attacks due to the misuse of LLMs, it is important to automate detection by leveraging the advanced capabilities of LLMs.   In this paper, we propose a novel system called ChatPhishDetector that utilizes LLMs to detect phishing sites. Our system involves leveraging a web crawler to gather information from websites, generating prompts for LLMs based on the crawled data, and then retrieving the detection results from the responses generated by the LLMs. The system enables us to detect multilingual phishing sites with high accuracy by identifying impersonated brands and social engineering techniques in the context of the entire website, without the need to train machine learning models. To evaluate the performance of our system, we conducted experiments on our own dataset and compared it with baseline systems and several LLMs. The experimental results using GPT-4V demonstrated outstanding performance, with a precision of 98.7% and a recall of 99.6%, outperforming the detection results of other LLMs and existing systems. These findings highlight the potential of LLMs for protecting users from online fraudulent activities and have important implications for enhancing cybersecurity measures.



## **31. PAL: Proxy-Guided Black-Box Attack on Large Language Models**

cs.CL

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09674v1) [paper-pdf](http://arxiv.org/pdf/2402.09674v1)

**Authors**: Chawin Sitawarin, Norman Mu, David Wagner, Alexandre Araujo

**Abstract**: Large Language Models (LLMs) have surged in popularity in recent months, but they have demonstrated concerning capabilities to generate harmful content when manipulated. While techniques like safety fine-tuning aim to minimize harmful use, recent works have shown that LLMs remain vulnerable to attacks that elicit toxic responses. In this work, we introduce the Proxy-Guided Attack on LLMs (PAL), the first optimization-based attack on LLMs in a black-box query-only setting. In particular, it relies on a surrogate model to guide the optimization and a sophisticated loss designed for real-world LLM APIs. Our attack achieves 84% attack success rate (ASR) on GPT-3.5-Turbo and 48% on Llama-2-7B, compared to 4% for the current state of the art. We also propose GCG++, an improvement to the GCG attack that reaches 94% ASR on white-box Llama-2-7B, and the Random-Search Attack on LLMs (RAL), a strong but simple baseline for query-based attacks. We believe the techniques proposed in this work will enable more comprehensive safety testing of LLMs and, in the long term, the development of better security guardrails. The code can be found at https://github.com/chawins/pal.



## **32. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

cs.RO

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09546v1) [paper-pdf](http://arxiv.org/pdf/2402.09546v1)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently shown impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the technology's widespread application in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Suffix (NPS) Attack that manipulates LLM-based navigation models by appending gradient-derived suffixes to the original navigational prompt, leading to incorrect actions. We conducted comprehensive experiments on an LLMs-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across three metrics in the face of both white-box and black-box attacks. These results highlight the generalizability and transferability of the NPS Attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, concentrating on navigation-relevant keywords to reduce the impact of adversarial suffixes. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.



## **33. Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey**

cs.CL

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09283v1) [paper-pdf](http://arxiv.org/pdf/2402.09283v1)

**Authors**: Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao

**Abstract**: Large Language Models (LLMs) are now commonplace in conversation applications. However, their risks of misuse for generating harmful responses have raised serious societal concerns and spurred recent research on LLM conversation safety. Therefore, in this survey, we provide a comprehensive overview of recent studies, covering three critical aspects of LLM conversation safety: attacks, defenses, and evaluations. Our goal is to provide a structured summary that enhances understanding of LLM conversation safety and encourages further investigation into this important subject. For easy reference, we have categorized all the studies mentioned in this survey according to our taxonomy, available at: https://github.com/niconi19/LLM-conversation-safety.



## **34. Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks**

cs.LG

29 pages

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09177v1) [paper-pdf](http://arxiv.org/pdf/2402.09177v1)

**Authors**: Yixin Cheng, Markos Georgopoulos, Volkan Cevher, Grigorios G. Chrysos

**Abstract**: Large Language Models (LLMs) are susceptible to Jailbreaking attacks, which aim to extract harmful information by subtly modifying the attack query. As defense mechanisms evolve, directly obtaining harmful information becomes increasingly challenging for Jailbreaking attacks. In this work, inspired by human practices of indirect context to elicit harmful information, we focus on a new attack form called Contextual Interaction Attack. The idea relies on the autoregressive nature of the generation process in LLMs. We contend that the prior context--the information preceding the attack query--plays a pivotal role in enabling potent Jailbreaking attacks. Specifically, we propose an approach that leverages preliminary question-answer pairs to interact with the LLM. By doing so, we guide the responses of the model toward revealing the 'desired' harmful information. We conduct experiments on four different LLMs and demonstrate the efficacy of this attack, which is black-box and can also transfer across LLMs. We believe this can lead to further developments and understanding of the context vector in LLMs.



## **35. Attacking Large Language Models with Projected Gradient Descent**

cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09154v1) [paper-pdf](http://arxiv.org/pdf/2402.09154v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.



## **36. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09063v1) [paper-pdf](http://arxiv.org/pdf/2402.09063v1)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.



## **37. Prompted Contextual Vectors for Spear-Phishing Detection**

cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08309v2) [paper-pdf](http://arxiv.org/pdf/2402.08309v2)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include an innovative document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.



## **38. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

cs.CR

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08983v1) [paper-pdf](http://arxiv.org/pdf/2402.08983v1)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.



## **39. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08679v1) [paper-pdf](http://arxiv.org/pdf/2402.08679v1)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on Large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent suffix attacks, but also allow us to address new controllable attack settings such as revising a user query adversarially with minimal paraphrasing, and inserting stealthy attacks in context with left-right-coherence. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.



## **40. Test-Time Backdoor Attacks on Multimodal Large Language Models**

cs.CL

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08577v1) [paper-pdf](http://arxiv.org/pdf/2402.08577v1)

**Authors**: Dong Lu, Tianyu Pang, Chao Du, Qian Liu, Xianjun Yang, Min Lin

**Abstract**: Backdoor attacks are commonly executed by contaminating training data, such that a trigger can activate predetermined harmful effects during the test phase. In this work, we present AnyDoor, a test-time backdoor attack against multimodal large language models (MLLMs), which involves injecting the backdoor into the textual modality using adversarial test images (sharing the same universal perturbation), without requiring access to or modification of the training data. AnyDoor employs similar techniques used in universal adversarial attacks, but distinguishes itself by its ability to decouple the timing of setup and activation of harmful effects. In our experiments, we validate the effectiveness of AnyDoor against popular MLLMs such as LLaVA-1.5, MiniGPT-4, InstructBLIP, and BLIP-2, as well as provide comprehensive ablation studies. Notably, because the backdoor is injected by a universal perturbation, AnyDoor can dynamically change its backdoor trigger prompts/harmful effects, exposing a new challenge for defending against backdoor attacks. Our project page is available at https://sail-sg.github.io/AnyDoor/.



## **41. Pandora: Jailbreak GPTs by Retrieval Augmented Generation Poisoning**

cs.CR

6 pages

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08416v1) [paper-pdf](http://arxiv.org/pdf/2402.08416v1)

**Authors**: Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tianwei Zhang, Yang Liu

**Abstract**: Large Language Models~(LLMs) have gained immense popularity and are being increasingly applied in various domains. Consequently, ensuring the security of these models is of paramount importance. Jailbreak attacks, which manipulate LLMs to generate malicious content, are recognized as a significant vulnerability. While existing research has predominantly focused on direct jailbreak attacks on LLMs, there has been limited exploration of indirect methods. The integration of various plugins into LLMs, notably Retrieval Augmented Generation~(RAG), which enables LLMs to incorporate external knowledge bases into their response generation such as GPTs, introduces new avenues for indirect jailbreak attacks.   To fill this gap, we investigate indirect jailbreak attacks on LLMs, particularly GPTs, introducing a novel attack vector named Retrieval Augmented Generation Poisoning. This method, Pandora, exploits the synergy between LLMs and RAG through prompt manipulation to generate unexpected responses. Pandora uses maliciously crafted content to influence the RAG process, effectively initiating jailbreak attacks. Our preliminary tests show that Pandora successfully conducts jailbreak attacks in four different scenarios, achieving higher success rates than direct attacks, with 64.3\% for GPT-3.5 and 34.8\% for GPT-4.



## **42. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

cs.CL

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2401.09002v2) [paper-pdf](http://arxiv.org/pdf/2401.09002v2)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.



## **43. PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining**

cs.CR

19 pages

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.09477v1) [paper-pdf](http://arxiv.org/pdf/2402.09477v1)

**Authors**: Mishaal Kazmi, Hadrien Lautraite, Alireza Akbari, Mauricio Soroco, Qiaoyue Tang, Tao Wang, Sébastien Gambs, Mathias Lécuyer

**Abstract**: We introduce a privacy auditing scheme for ML models that relies on membership inference attacks using generated data as "non-members". This scheme, which we call PANORAMIA, quantifies the privacy leakage for large-scale ML models without control of the training process or model re-training and only requires access to a subset of the training data. To demonstrate its applicability, we evaluate our auditing scheme across multiple ML domains, ranging from image and tabular data classification to large-scale language models.



## **44. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2309.02705v3) [paper-pdf](http://arxiv.org/pdf/2309.02705v3)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.



## **45. PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models**

cs.CR

Code is available at https://github.com/sleeepeer/PoisonedRAG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07867v1) [paper-pdf](http://arxiv.org/pdf/2402.07867v1)

**Authors**: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia

**Abstract**: Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate those limitations. In particular, given a question, RAG retrieves relevant knowledge from a knowledge database to augment the input of the LLM. For instance, the retrieved knowledge could be a set of top-k texts that are most semantically similar to the given question when the knowledge database contains millions of texts collected from Wikipedia. As a result, the LLM could utilize the retrieved knowledge as the context to generate an answer for the given question. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. Particularly, we propose PoisonedRAG , a set of knowledge poisoning attacks to RAG, where an attacker could inject a few poisoned texts into the knowledge database such that the LLM generates an attacker-chosen target answer for an attacker-chosen target question. We formulate knowledge poisoning attacks as an optimization problem, whose solution is a set of poisoned texts. Depending on the background knowledge (e.g., black-box and white-box settings) of an attacker on the RAG, we propose two solutions to solve the optimization problem, respectively. Our results on multiple benchmark datasets and LLMs show our attacks could achieve 90% attack success rates when injecting 5 poisoned texts for each target question into a database with millions of texts. We also evaluate recent defenses and our results show they are insufficient to defend against our attacks, highlighting the need for new defenses.



## **46. Do Membership Inference Attacks Work on Large Language Models?**

cs.CL

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07841v1) [paper-pdf](http://arxiv.org/pdf/2402.07841v1)

**Authors**: Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi

**Abstract**: Membership inference attacks (MIAs) attempt to predict whether a particular datapoint is a member of a target model's training data. Despite extensive research on traditional machine learning models, there has been limited work studying MIA on the pre-training data of large language models (LLMs). We perform a large-scale evaluation of MIAs over a suite of language models (LMs) trained on the Pile, ranging from 160M to 12B parameters. We find that MIAs barely outperform random guessing for most settings across varying LLM sizes and domains. Our further analyses reveal that this poor performance can be attributed to (1) the combination of a large dataset and few training iterations, and (2) an inherently fuzzy boundary between members and non-members. We identify specific settings where LLMs have been shown to be vulnerable to membership inference and show that the apparent success in such settings can be attributed to a distribution shift, such as when members and non-members are drawn from the seemingly identical domain but with different temporal ranges. We release our code and data as a unified benchmark package that includes all existing MIAs, supporting future work.



## **47. Universal Jailbreak Backdoors from Poisoned Human Feedback**

cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2311.14455v3) [paper-pdf](http://arxiv.org/pdf/2311.14455v3)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.



## **48. Large Language Models are Few-shot Generators: Proposing Hybrid Prompt Algorithm To Generate Webshell Escape Samples**

cs.CR

13 pages, 16 figures

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07408v1) [paper-pdf](http://arxiv.org/pdf/2402.07408v1)

**Authors**: Mingrui Ma, Lansheng Han, Chunjie Zhou

**Abstract**: The frequent occurrence of cyber-attacks has made webshell attacks and defense gradually become a research hotspot in the field of network security. However, the lack of publicly available benchmark datasets and the over-reliance on manually defined rules for webshell escape sample generation have slowed down the progress of research related to webshell escape sample generation strategies and artificial intelligence-based webshell detection algorithms. To address the drawbacks of weak webshell sample escape capabilities, the lack of webshell datasets with complex malicious features, and to promote the development of webshell detection technology, we propose the Hybrid Prompt algorithm for webshell escape sample generation with the help of large language models. As a prompt algorithm specifically developed for webshell sample generation, the Hybrid Prompt algorithm not only combines various prompt ideas including Chain of Thought, Tree of Thought, but also incorporates various components such as webshell hierarchical module and few-shot example to facilitate the LLM in learning and reasoning webshell escape strategies. Experimental results show that the Hybrid Prompt algorithm can work with multiple LLMs with excellent code reasoning ability to generate high-quality webshell samples with high Escape Rate (88.61% with GPT-4 model on VIRUSTOTAL detection engine) and Survival Rate (54.98% with GPT-4 model).



## **49. All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**

cs.CL

12 pages, 4 figures, 3 tables

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2401.09798v3) [paper-pdf](http://arxiv.org/pdf/2401.09798v3)

**Authors**: Kazuhiro Takemoto

**Abstract**: Large Language Models (LLMs), such as ChatGPT, encounter `jailbreak' challenges, wherein safeguards are circumvented to generate ethically harmful prompts. This study introduces a straightforward black-box method for efficiently crafting jailbreak prompts, addressing the significant complexity and computational costs associated with conventional methods. Our technique iteratively transforms harmful prompts into benign expressions directly utilizing the target LLM, predicated on the hypothesis that LLMs can autonomously generate expressions that evade safeguards. Through experiments conducted with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, our method consistently achieved an attack success rate exceeding 80% within an average of five iterations for forbidden questions and proved robust against model updates. The jailbreak prompts generated were not only naturally-worded and succinct but also challenging to defend against. These findings suggest that the creation of effective jailbreak prompts is less complex than previously believed, underscoring the heightened risk posed by black-box jailbreak attacks.



## **50. Whispers in the Machine: Confidentiality in LLM-integrated Systems**

cs.CR

**SubmitDate**: 2024-02-10    [abs](http://arxiv.org/abs/2402.06922v1) [paper-pdf](http://arxiv.org/pdf/2402.06922v1)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.   In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization across attack strategies. Building on this analysis, we propose a method for robustness fine-tuning, inspired by adversarial training. This approach is effective in lowering the success rate of attackers and in improving the system's resilience against unknown attacks.



