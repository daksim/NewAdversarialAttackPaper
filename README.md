# Latest Adversarial Attack Papers
**update at 2024-08-22 17:18:42**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Competence-Based Analysis of Language Models**

cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2303.00333v4) [paper-pdf](http://arxiv.org/pdf/2303.00333v4)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain and predict behaviors across these tasks.



## **2. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11749v1) [paper-pdf](http://arxiv.org/pdf/2408.11749v1)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.



## **3. First line of defense: A robust first layer mitigates adversarial attacks**

cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11680v1) [paper-pdf](http://arxiv.org/pdf/2408.11680v1)

**Authors**: Janani Suresh, Nancy Nayak, Sheetal Kalyani

**Abstract**: Adversarial training (AT) incurs significant computational overhead, leading to growing interest in designing inherently robust architectures. We demonstrate that a carefully designed first layer of the neural network can serve as an implicit adversarial noise filter (ANF). This filter is created using a combination of large kernel size, increased convolution filters, and a maxpool operation. We show that integrating this filter as the first layer in architectures such as ResNet, VGG, and EfficientNet results in adversarially robust networks. Our approach achieves higher adversarial accuracies than existing natively robust architectures without AT and is competitive with adversarial-trained architectures across a wide range of datasets. Supporting our findings, we show that (a) the decision regions for our method have better margins, (b) the visualized loss surfaces are smoother, (c) the modified peak signal-to-noise ratio (mPSNR) values at the output of the ANF are higher, (d) high-frequency components are more attenuated, and (e) architectures incorporating ANF exhibit better denoising in Gaussian noise compared to baseline architectures. Code for all our experiments are available at \url{https://github.com/janani-suresh-97/first-line-defence.git}.



## **4. Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection**

cs.CV

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11408v1) [paper-pdf](http://arxiv.org/pdf/2408.11408v1)

**Authors**: Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun

**Abstract**: Multi-View Diffusion Models (MVDMs) enable remarkable improvements in the field of 3D geometric reconstruction, but the issue regarding intellectual property has received increasing attention due to unauthorized imitation. Recently, some works have utilized adversarial attacks to protect copyright. However, all these works focus on single-image generation tasks which only need to consider the inner feature of images. Previous methods are inefficient in attacking MVDMs because they lack the consideration of disrupting the geometric and visual consistency among the generated multi-view images. This paper is the first to address the intellectual property infringement issue arising from MVDMs. Accordingly, we propose a novel latent feature and attention dual erasure attack to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously. The experiments conducted on SOTA MVDMs indicate that our approach achieves superior performances in terms of attack effectiveness, transferability, and robustness against defense methods. Therefore, this paper provides an efficient solution to protect 3D assets from MVDMs-based 3D geometry reconstruction.



## **5. AntifakePrompt: Prompt-Tuned Vision-Language Models are Fake Image Detectors**

cs.CV

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2310.17419v3) [paper-pdf](http://arxiv.org/pdf/2310.17419v3)

**Authors**: You-Ming Chang, Chen Yeh, Wei-Chen Chiu, Ning Yu

**Abstract**: Deep generative models can create remarkably photorealistic fake images while raising concerns about misinformation and copyright infringement, known as deepfake threats. Deepfake detection technique is developed to distinguish between real and fake images, where the existing methods typically learn classifiers in the image domain or various feature domains. However, the generalizability of deepfake detection against emerging and more advanced generative models remains challenging. In this paper, being inspired by the zero-shot advantages of Vision-Language Models (VLMs), we propose a novel approach called AntifakePrompt, using VLMs (e.g., InstructBLIP) and prompt tuning techniques to improve the deepfake detection accuracy over unseen data. We formulate deepfake detection as a visual question answering problem, and tune soft prompts for InstructBLIP to answer the real/fake information of a query image. We conduct full-spectrum experiments on datasets from a diversity of 3 held-in and 20 held-out generative models, covering modern text-to-image generation, image editing and adversarial image attacks. These testing datasets provide useful benchmarks in the realm of deepfake detection for further research. Moreover, results demonstrate that (1) the deepfake detection accuracy can be significantly and consistently improved (from 71.06% to 92.11%, in average accuracy over unseen domains) using pretrained vision-language models with prompt tuning; (2) our superior performance is at less cost of training data and trainable parameters, resulting in an effective and efficient solution for deepfake detection. Code and models can be found at https://github.com/nctu-eva-lab/AntifakePrompt.



## **6. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

physics.soc-ph

17 pages, 4 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.19692v3) [paper-pdf](http://arxiv.org/pdf/2406.19692v3)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.



## **7. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

cs.LG

33 pages

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.11463v2) [paper-pdf](http://arxiv.org/pdf/2407.11463v2)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models by causing incorrect predictions through imperceptible perturbations to the input data. While these attacks have been extensively studied in unstructured data like images, applying them to tabular data, poses new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ from the image data. To account for this distinction, it is necessary to establish tailored imperceptibility criteria specific to tabular data. However, there is currently a lack of standardised metrics for assessing the imperceptibility of adversarial attacks on tabular data. To address this gap, we propose a set of key properties and corresponding metrics designed to comprehensively characterise imperceptible adversarial attacks on tabular data. These are: proximity to the original input, sparsity of altered features, deviation from the original data distribution, sensitivity in perturbing features with narrow distribution, immutability of certain features that should remain unchanged, feasibility of specific feature values that should not go beyond valid practical ranges, and feature interdependencies capturing complex relationships between data attributes. We evaluate the imperceptibility of five adversarial attacks, including both bounded attacks and unbounded attacks, on tabular data using the proposed imperceptibility metrics. The results reveal a trade-off between the imperceptibility and effectiveness of these attacks. The study also identifies limitations in current attack algorithms, offering insights that can guide future research in the area. The findings gained from this empirical analysis provide valuable direction for enhancing the design of adversarial attack algorithms, thereby advancing adversarial machine learning on tabular data.



## **8. Unlocking Adversarial Suffix Optimization Without Affirmative Phrases: Efficient Black-box Jailbreaking via LLM as Optimizer**

cs.AI

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11313v1) [paper-pdf](http://arxiv.org/pdf/2408.11313v1)

**Authors**: Weipeng Jiang, Zhenting Wang, Juan Zhai, Shiqing Ma, Zhengyu Zhao, Chao Shen

**Abstract**: Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.



## **9. EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models**

cs.AI

19 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11308v1) [paper-pdf](http://arxiv.org/pdf/2408.11308v1)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. Built upon this idea, we introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85\% in comparison with 50\% for the present SOTAs, with minimal impact on the utility and effectiveness of LLMs.



## **10. Correlation Analysis of Adversarial Attack in Time Series Classification**

cs.LG

15 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11264v1) [paper-pdf](http://arxiv.org/pdf/2408.11264v1)

**Authors**: Zhengyang Li, Wenhao Liang, Chang Dong, Weitong Chen, Dong Huang

**Abstract**: This study investigates the vulnerability of time series classification models to adversarial attacks, with a focus on how these models process local versus global information under such conditions. By leveraging the Normalized Auto Correlation Function (NACF), an exploration into the inclination of neural networks is conducted. It is demonstrated that regularization techniques, particularly those employing Fast Fourier Transform (FFT) methods and targeting frequency components of perturbations, markedly enhance the effectiveness of attacks. Meanwhile, the defense strategies, like noise introduction and Gaussian filtering, are shown to significantly lower the Attack Success Rate (ASR), with approaches based on noise introducing notably effective in countering high-frequency distortions. Furthermore, models designed to prioritize global information are revealed to possess greater resistance to adversarial manipulations. These results underline the importance of designing attack and defense mechanisms, informed by frequency domain analysis, as a means to considerably reinforce the resilience of neural network models against adversarial threats.



## **11. Revisiting Min-Max Optimization Problem in Adversarial Training**

cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11218v1) [paper-pdf](http://arxiv.org/pdf/2408.11218v1)

**Authors**: Sina Hajer Ahmadi, Hassan Bahrami

**Abstract**: The rise of computer vision applications in the real world puts the security of the deep neural networks at risk. Recent works demonstrate that convolutional neural networks are susceptible to adversarial examples - where the input images look similar to the natural images but are classified incorrectly by the model. To provide a rebuttal to this problem, we propose a new method to build robust deep neural networks against adversarial attacks by reformulating the saddle point optimization problem in \cite{madry2017towards}. Our proposed method offers significant resistance and a concrete security guarantee against multiple adversaries. The goal of this paper is to act as a stepping stone for a new variation of deep learning models which would lead towards fully robust deep learning models.



## **12. GAIM: Attacking Graph Neural Networks via Adversarial Influence Maximization**

cs.LG

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10948v1) [paper-pdf](http://arxiv.org/pdf/2408.10948v1)

**Authors**: Xiaodong Yang, Xiaoting Li, Huiyuan Chen, Yiwei Cai

**Abstract**: Recent studies show that well-devised perturbations on graph structures or node features can mislead trained Graph Neural Network (GNN) models. However, these methods often overlook practical assumptions, over-rely on heuristics, or separate vital attack components. In response, we present GAIM, an integrated adversarial attack method conducted on a node feature basis while considering the strict black-box setting. Specifically, we define an adversarial influence function to theoretically assess the adversarial impact of node perturbations, thereby reframing the GNN attack problem into the adversarial influence maximization problem. In our approach, we unify the selection of the target node and the construction of feature perturbations into a single optimization problem, ensuring a unique and consistent feature perturbation for each target node. We leverage a surrogate model to transform this problem into a solvable linear programming task, streamlining the optimization process. Moreover, we extend our method to accommodate label-oriented attacks, broadening its applicability. Thorough evaluations on five benchmark datasets across three popular models underscore the effectiveness of our method in both untargeted and label-oriented targeted attacks. Through comprehensive analysis and ablation studies, we demonstrate the practical value and efficacy inherent to our design choices.



## **13. A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

cs.CV

21 pages, 7 figures, 10 tables

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10901v1) [paper-pdf](http://arxiv.org/pdf/2408.10901v1)

**Authors**: Zhongliang Guo, Lei Fang, Jingyu Lin, Yifei Qian, Shuai Zhao, Zeyu Wang, Junhao Dong, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in generative AI, particularly Latent Diffusion Models (LDMs), have revolutionized image synthesis and manipulation. However, these generative techniques raises concerns about data misappropriation and intellectual property infringement. Adversarial attacks on machine learning models have been extensively studied, and a well-established body of research has extended these techniques as a benign metric to prevent the underlying misuse of generative AI. Current approaches to safeguarding images from manipulation by LDMs are limited by their reliance on model-specific knowledge and their inability to significantly degrade semantic quality of generated images. In response to these shortcomings, we propose the Posterior Collapse Attack (PCA) based on the observation that VAEs suffer from posterior collapse during training. Our method minimizes dependence on the white-box information of target models to get rid of the implicit reliance on model-specific knowledge. By accessing merely a small amount of LDM parameters, in specific merely the VAE encoder of LDMs, our method causes a substantial semantic collapse in generation quality, particularly in perceptual consistency, and demonstrates strong transferability across various model architectures. Experimental results show that PCA achieves superior perturbation effects on image generation of LDMs with lower runtime and VRAM. Our method outperforms existing techniques, offering a more robust and generalizable solution that is helpful in alleviating the socio-technical challenges posed by the rapidly evolving landscape of generative AI.



## **14. Towards Efficient Formal Verification of Spiking Neural Network**

cs.AI

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10900v1) [paper-pdf](http://arxiv.org/pdf/2408.10900v1)

**Authors**: Baekryun Seong, Jieung Kim, Sang-Ki Ko

**Abstract**: Recently, AI research has primarily focused on large language models (LLMs), and increasing accuracy often involves scaling up and consuming more power. The power consumption of AI has become a significant societal issue; in this context, spiking neural networks (SNNs) offer a promising solution. SNNs operate event-driven, like the human brain, and compress information temporally. These characteristics allow SNNs to significantly reduce power consumption compared to perceptron-based artificial neural networks (ANNs), highlighting them as a next-generation neural network technology. However, societal concerns regarding AI go beyond power consumption, with the reliability of AI models being a global issue. For instance, adversarial attacks on AI models are a well-studied problem in the context of traditional neural networks. Despite their importance, the stability and property verification of SNNs remains in the early stages of research. Most SNN verification methods are time-consuming and barely scalable, making practical applications challenging. In this paper, we introduce temporal encoding to achieve practical performance in verifying the adversarial robustness of SNNs. We conduct a theoretical analysis of this approach and demonstrate its success in verifying SNNs at previously unmanageable scales. Our contribution advances SNN verification to a practical level, facilitating the safer application of SNNs.



## **15. Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2004.12571v3) [paper-pdf](http://arxiv.org/pdf/2004.12571v3)

**Authors**: Xinjian Luo, Xianglong Zhang

**Abstract**: Federated learning (FL) is a decentralized model training framework that aims to merge isolated data islands while maintaining data privacy. However, recent studies have revealed that Generative Adversarial Network (GAN) based attacks can be employed in FL to learn the distribution of private datasets and reconstruct recognizable images. In this paper, we exploit defenses against GAN-based attacks in FL and propose a framework, Anti-GAN, to prevent attackers from learning the real distribution of the victim's data. The core idea of Anti-GAN is to manipulate the visual features of private training images to make them indistinguishable to human eyes even restored by attackers. Specifically, Anti-GAN projects the private dataset onto a GAN's generator and combines the generated fake images with the actual images to create the training dataset, which is then used for federated model training. The experimental results demonstrate that Anti-GAN is effective in preventing attackers from learning the distribution of private images while causing minimal harm to the accuracy of the federated model.



## **16. Honeyquest: Rapidly Measuring the Enticingness of Cyber Deception Techniques with Code-based Questionnaires**

cs.CR

to be published in the 27th International Symposium on Research in  Attacks, Intrusions and Defenses (RAID 2024), dataset and source code  available at https://github.com/dynatrace-oss/honeyquest

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10796v1) [paper-pdf](http://arxiv.org/pdf/2408.10796v1)

**Authors**: Mario Kahlhofer, Stefan Achleitner, Stefan Rass, René Mayrhofer

**Abstract**: Fooling adversaries with traps such as honeytokens can slow down cyber attacks and create strong indicators of compromise. Unfortunately, cyber deception techniques are often poorly specified. Also, realistically measuring their effectiveness requires a well-exposed software system together with a production-ready implementation of these techniques. This makes rapid prototyping challenging. Our work translates 13 previously researched and 12 self-defined techniques into a high-level, machine-readable specification. Our open-source tool, Honeyquest, allows researchers to quickly evaluate the enticingness of deception techniques without implementing them. We test the enticingness of 25 cyber deception techniques and 19 true security risks in an experiment with 47 humans. We successfully replicate the goals of previous work with many consistent findings, but without a time-consuming implementation of these techniques on real computer systems. We provide valuable insights for the design of enticing deception and also show that the presence of cyber deception can significantly reduce the risk that adversaries will find a true security risk by about 22% on average.



## **17. Adversarial Attack for Explanation Robustness of Rationalization Models**

cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10795v1) [paper-pdf](http://arxiv.org/pdf/2408.10795v1)

**Authors**: Yuankai Zhang, Lingxiao Kong, Haozhao Wang, Ruixuan Li, Jun Wang, Yuhua Li, Wei Liu

**Abstract**: Rationalization models, which select a subset of input text as rationale-crucial for humans to understand and trust predictions-have recently emerged as a prominent research area in eXplainable Artificial Intelligence. However, most of previous studies mainly focus on improving the quality of the rationale, ignoring its robustness to malicious attack. Specifically, whether the rationalization models can still generate high-quality rationale under the adversarial attack remains unknown. To explore this, this paper proposes UAT2E, which aims to undermine the explainability of rationalization models without altering their predictions, thereby eliciting distrust in these models from human users. UAT2E employs the gradient-based search on triggers and then inserts them into the original input to conduct both the non-target and target attack. Experimental results on five datasets reveal the vulnerability of rationalization models in terms of explanation, where they tend to select more meaningless tokens under attacks. Based on this, we make a series of recommendations for improving rationalization models in terms of explanation.



## **18. SAM Meets UAP: Attacking Segment Anything Model With Universal Adversarial Perturbation**

cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2310.12431v2) [paper-pdf](http://arxiv.org/pdf/2310.12431v2)

**Authors**: Dongshen Han, Chaoning Zhang, Sheng Zheng, Chang Lu, Yang Yang, Heng Tao Shen

**Abstract**: As Segment Anything Model (SAM) becomes a popular foundation model in computer vision, its adversarial robustness has become a concern that cannot be ignored. This works investigates whether it is possible to attack SAM with image-agnostic Universal Adversarial Perturbation (UAP). In other words, we seek a single perturbation that can fool the SAM to predict invalid masks for most (if not all) images. We demonstrate convetional image-centric attack framework is effective for image-independent attacks but fails for universal adversarial attack. To this end, we propose a novel perturbation-centric framework that results in a UAP generation method based on self-supervised contrastive learning (CL), where the UAP is set to the anchor sample and the positive sample is augmented from the UAP. The representations of negative samples are obtained from the image encoder in advance and saved in a memory bank. The effectiveness of our proposed CL-based UAP generation method is validated by both quantitative and qualitative results. On top of the ablation study to understand various components in our proposed method, we shed light on the roles of positive and negative samples in making the generated UAP effective for attacking SAM.



## **19. Security Assessment of Hierarchical Federated Deep Learning**

cs.LG

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10752v1) [paper-pdf](http://arxiv.org/pdf/2408.10752v1)

**Authors**: D Alqattan, R Sun, H Liang, G Nicosia, V Snasel, R Ranjan, V Ojha

**Abstract**: Hierarchical federated learning (HFL) is a promising distributed deep learning model training paradigm, but it has crucial security concerns arising from adversarial attacks. This research investigates and assesses the security of HFL using a novel methodology by focusing on its resilience against adversarial attacks inference-time and training-time. Through a series of extensive experiments across diverse datasets and attack scenarios, we uncover that HFL demonstrates robustness against untargeted training-time attacks due to its hierarchical structure. However, targeted attacks, particularly backdoor attacks, exploit this architecture, especially when malicious clients are positioned in the overlapping coverage areas of edge servers. Consequently, HFL shows a dual nature in its resilience, showcasing its capability to recover from attacks thanks to its hierarchical aggregation that strengthens its suitability for adversarial training, thereby reinforcing its resistance against inference-time attacks. These insights underscore the necessity for balanced security strategies in HFL systems, leveraging their inherent strengths while effectively mitigating vulnerabilities.



## **20. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10738v1) [paper-pdf](http://arxiv.org/pdf/2408.10738v1)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also encounter notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the top k relevant items from offline knowledge bases, utilizing all available information from a webpage, including logos, HTML, and URLs. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.



## **21. Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique**

cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10701v1) [paper-pdf](http://arxiv.org/pdf/2408.10701v1)

**Authors**: Tej Deep Pala, Vernon Y. H. Toh, Rishabh Bhardwaj, Soujanya Poria

**Abstract**: In today's era, where large language models (LLMs) are integrated into numerous real-world applications, ensuring their safety and robustness is crucial for responsible AI usage. Automated red-teaming methods play a key role in this process by generating adversarial attacks to identify and mitigate potential vulnerabilities in these models. However, existing methods often struggle with slow performance, limited categorical diversity, and high resource demands. While Rainbow Teaming, a recent approach, addresses the diversity challenge by framing adversarial prompt generation as a quality-diversity search, it remains slow and requires a large fine-tuned mutator for optimal performance. To overcome these limitations, we propose Ferret, a novel approach that builds upon Rainbow Teaming by generating multiple adversarial prompt mutations per iteration and using a scoring function to rank and select the most effective adversarial prompt. We explore various scoring functions, including reward models, Llama Guard, and LLM-as-a-judge, to rank adversarial mutations based on their potential harm to improve the efficiency of the search for harmful mutations. Our results demonstrate that Ferret, utilizing a reward model as a scoring function, improves the overall attack success rate (ASR) to 95%, which is 46% higher than Rainbow Teaming. Additionally, Ferret reduces the time needed to achieve a 90% ASR by 15.2% compared to the baseline and generates adversarial prompts that are transferable i.e. effective on other LLMs of larger size. Our codes are available at https://github.com/declare-lab/ferret.



## **22. MsMemoryGAN: A Multi-scale Memory GAN for Palm-vein Adversarial Purification**

cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10694v1) [paper-pdf](http://arxiv.org/pdf/2408.10694v1)

**Authors**: Huafeng Qin, Yuming Fu, Huiyan Zhang, Mounim A. El-Yacoubi, Xinbo Gao, Qun Song, Jun Wang

**Abstract**: Deep neural networks have recently achieved promising performance in the vein recognition task and have shown an increasing application trend, however, they are prone to adversarial perturbation attacks by adding imperceptible perturbations to the input, resulting in making incorrect recognition. To address this issue, we propose a novel defense model named MsMemoryGAN, which aims to filter the perturbations from adversarial samples before recognition. First, we design a multi-scale autoencoder to achieve high-quality reconstruction and two memory modules to learn the detailed patterns of normal samples at different scales. Second, we investigate a learnable metric in the memory module to retrieve the most relevant memory items to reconstruct the input image. Finally, the perceptional loss is combined with the pixel loss to further enhance the quality of the reconstructed image. During the training phase, the MsMemoryGAN learns to reconstruct the input by merely using fewer prototypical elements of the normal patterns recorded in the memory. At the testing stage, given an adversarial sample, the MsMemoryGAN retrieves its most relevant normal patterns in memory for the reconstruction. Perturbations in the adversarial sample are usually not reconstructed well, resulting in purifying the input from adversarial perturbations. We have conducted extensive experiments on two public vein datasets under different adversarial attack methods to evaluate the performance of the proposed approach. The experimental results show that our approach removes a wide variety of adversarial perturbations, allowing vein classifiers to achieve the highest recognition accuracy.



## **23. Towards Robust Knowledge Unlearning: An Adversarial Framework for Assessing and Improving Unlearning Robustness in Large Language Models**

cs.CL

13 pages

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10682v1) [paper-pdf](http://arxiv.org/pdf/2408.10682v1)

**Authors**: Hongbang Yuan, Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: LLM have achieved success in many fields but still troubled by problematic content in the training corpora. LLM unlearning aims at reducing their influence and avoid undesirable behaviours. However, existing unlearning methods remain vulnerable to adversarial queries and the unlearned knowledge resurfaces after the manually designed attack queries. As part of a red-team effort to proactively assess the vulnerabilities of unlearned models, we design Dynamic Unlearning Attack (DUA), a dynamic and automated framework to attack these models and evaluate their robustness. It optimizes adversarial suffixes to reintroduce the unlearned knowledge in various scenarios. We find that unlearned knowledge can be recovered in $55.2\%$ of the questions, even without revealing the unlearned model's parameters. In response to this vulnerability, we propose Latent Adversarial Unlearning (LAU), a universal framework that effectively enhances the robustness of the unlearned process. It formulates the unlearning process as a min-max optimization problem and resolves it through two stages: an attack stage, where perturbation vectors are trained and added to the latent space of LLMs to recover the unlearned knowledge, and a defense stage, where previously trained perturbation vectors are used to enhance unlearned model's robustness. With our LAU framework, we obtain two robust unlearning methods, AdvGA and AdvNPO. We conduct extensive experiments across multiple unlearning benchmarks and various models, and demonstrate that they improve the unlearning effectiveness by over $53.5\%$, cause only less than a $11.6\%$ reduction in neighboring knowledge, and have almost no impact on the model's general capabilities.



## **24. Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification**

cs.CR

Under review

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10673v1) [paper-pdf](http://arxiv.org/pdf/2408.10673v1)

**Authors**: Hanrui Wang, Ruoxi Sun, Cunjian Chen, Minhui Xue, Lay-Ki Soon, Shuo Wang, Zhe Jin

**Abstract**: Face authentication systems have brought significant convenience and advanced developments, yet they have become unreliable due to their sensitivity to inconspicuous perturbations, such as adversarial attacks. Existing defenses often exhibit weaknesses when facing various attack algorithms and adaptive attacks or compromise accuracy for enhanced security. To address these challenges, we have developed a novel and highly efficient non-deep-learning-based image filter called the Iterative Window Mean Filter (IWMF) and proposed a new framework for adversarial purification, named IWMF-Diff, which integrates IWMF and denoising diffusion models. These methods can function as pre-processing modules to eliminate adversarial perturbations without necessitating further modifications or retraining of the target system. We demonstrate that our proposed methodologies fulfill four critical requirements: preserved accuracy, improved security, generalizability to various threats in different settings, and better resistance to adaptive attacks. This performance surpasses that of the state-of-the-art adversarial purification method, DiffPure.



## **25. Accelerating the Surrogate Retraining for Poisoning Attacks against Recommender Systems**

cs.IR

Accepted by RecSys 2024

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10666v1) [paper-pdf](http://arxiv.org/pdf/2408.10666v1)

**Authors**: Yunfan Wu, Qi Cao, Shuchang Tao, Kaike Zhang, Fei Sun, Huawei Shen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data poisoning attacks, where adversaries inject carefully crafted fake user interactions into the training data of recommenders to promote target items. Current attack methods involve iteratively retraining a surrogate recommender on the poisoned data with the latest fake users to optimize the attack. However, this repetitive retraining is highly time-consuming, hindering the efficient assessment and optimization of fake users. To mitigate this computational bottleneck and develop a more effective attack in an affordable time, we analyze the retraining process and find that a change in the representation of one user/item will cause a cascading effect through the user-item interaction graph. Under theoretical guidance, we introduce \emph{Gradient Passing} (GP), a novel technique that explicitly passes gradients between interacted user-item pairs during backpropagation, thereby approximating the cascading effect and accelerating retraining. With just a single update, GP can achieve effects comparable to multiple original training iterations. Under the same number of retraining epochs, GP enables a closer approximation of the surrogate recommender to the victim. This more accurate approximation provides better guidance for optimizing fake users, ultimately leading to enhanced data poisoning attacks. Extensive experiments on real-world datasets demonstrate the efficiency and effectiveness of our proposed GP.



## **26. Privacy-preserving Universal Adversarial Defense for Black-box Models**

cs.LG

12 pages, 9 figures

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10647v1) [paper-pdf](http://arxiv.org/pdf/2408.10647v1)

**Authors**: Qiao Li, Cong Wu, Jing Chen, Zijun Zhang, Kun He, Ruiying Du, Xinxin Wang, Qingchuang Zhao, Yang Liu

**Abstract**: Deep neural networks (DNNs) are increasingly used in critical applications such as identity authentication and autonomous driving, where robustness against adversarial attacks is crucial. These attacks can exploit minor perturbations to cause significant prediction errors, making it essential to enhance the resilience of DNNs. Traditional defense methods often rely on access to detailed model information, which raises privacy concerns, as model owners may be reluctant to share such data. In contrast, existing black-box defense methods fail to offer a universal defense against various types of adversarial attacks. To address these challenges, we introduce DUCD, a universal black-box defense method that does not require access to the target model's parameters or architecture. Our approach involves distilling the target model by querying it with data, creating a white-box surrogate while preserving data privacy. We further enhance this surrogate model using a certified defense based on randomized smoothing and optimized noise selection, enabling robust defense against a broad range of adversarial attacks. Comparative evaluations between the certified defenses of the surrogate and target models demonstrate the effectiveness of our approach. Experiments on multiple image classification datasets show that DUCD not only outperforms existing black-box defenses but also matches the accuracy of white-box defenses, all while enhancing data privacy and reducing the success rate of membership inference attacks.



## **27. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2405.09863v3) [paper-pdf](http://arxiv.org/pdf/2405.09863v3)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.



## **28. Prompt-Agnostic Adversarial Perturbation for Customized Diffusion Models**

cs.CV

33 pages, 14 figures, under review

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10571v1) [paper-pdf](http://arxiv.org/pdf/2408.10571v1)

**Authors**: Cong Wan, Yuhang He, Xiang Song, Yihong Gong

**Abstract**: Diffusion models have revolutionized customized text-to-image generation, allowing for efficient synthesis of photos from personal data with textual descriptions. However, these advancements bring forth risks including privacy breaches and unauthorized replication of artworks. Previous researches primarily center around using prompt-specific methods to generate adversarial examples to protect personal images, yet the effectiveness of existing methods is hindered by constrained adaptability to different prompts. In this paper, we introduce a Prompt-Agnostic Adversarial Perturbation (PAP) method for customized diffusion models. PAP first models the prompt distribution using a Laplace Approximation, and then produces prompt-agnostic perturbations by maximizing a disturbance expectation based on the modeled distribution. This approach effectively tackles the prompt-agnostic attacks, leading to improved defense stability. Extensive experiments in face privacy and artistic style protection, demonstrate the superior generalization of our method in comparison to existing techniques.



## **29. Enhancing Adversarial Transferability with Adversarial Weight Tuning**

cs.CR

13 pages

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.09469v2) [paper-pdf](http://arxiv.org/pdf/2408.09469v2)

**Authors**: Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, Shouling Ji

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs) that mislead the model while appearing benign to human observers. A critical concern is the transferability of AEs, which enables black-box attacks without direct access to the target model. However, many previous attacks have failed to explain the intrinsic mechanism of adversarial transferability. In this paper, we rethink the property of transferable AEs and reformalize the formulation of transferability. Building on insights from this mechanism, we analyze the generalization of AEs across models with different architectures and prove that we can find a local perturbation to mitigate the gap between surrogate and target models. We further establish the inner connections between model smoothness and flat local maxima, both of which contribute to the transferability of AEs. Further, we propose a new adversarial attack algorithm, \textbf{A}dversarial \textbf{W}eight \textbf{T}uning (AWT), which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs. Extensive experiments on a variety of models with different architectures on ImageNet demonstrate that AWT yields superior performance over other attacks, with an average increase of nearly 5\% and 10\% attack success rates on CNN-based and Transformer-based models, respectively, compared to state-of-the-art attacks.



## **30. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

Accepted by Journal of Machine Learning Research (JMLR); code:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2312.07910v3) [paper-pdf](http://arxiv.org/pdf/2312.07910v3)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **31. Fight Perturbations with Perturbations: Defending Adversarial Attacks via Neuron Influence**

cs.CV

Final version. Accepted to IEEE Transactions on Dependable and Secure  Computing

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2112.13060v3) [paper-pdf](http://arxiv.org/pdf/2112.13060v3)

**Authors**: Ruoxi Chen, Haibo Jin, Haibin Zheng, Jinyin Chen, Zhenguang Liu

**Abstract**: The vulnerabilities of deep learning models towards adversarial attacks have attracted increasing attention, especially when models are deployed in security-critical domains. Numerous defense methods, including reactive and proactive ones, have been proposed for model robustness improvement. Reactive defenses, such as conducting transformations to remove perturbations, usually fail to handle large perturbations. The proactive defenses that involve retraining, suffer from the attack dependency and high computation cost. In this paper, we consider defense methods from the general effect of adversarial attacks that take on neurons inside the model. We introduce the concept of neuron influence, which can quantitatively measure neurons' contribution to correct classification. Then, we observe that almost all attacks fool the model by suppressing neurons with larger influence and enhancing those with smaller influence. Based on this, we propose \emph{Neuron-level Inverse Perturbation} (NIP), a novel defense against general adversarial attacks. It calculates neuron influence from benign examples and then modifies input examples by generating inverse perturbations that can in turn strengthen neurons with larger influence and weaken those with smaller influence.



## **32. Detecting Adversarial Attacks in Semantic Segmentation via Uncertainty Estimation: A Deep Analysis**

cs.CV

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10021v1) [paper-pdf](http://arxiv.org/pdf/2408.10021v1)

**Authors**: Kira Maag, Roman Resner, Asja Fischer

**Abstract**: Deep neural networks have demonstrated remarkable effectiveness across a wide range of tasks such as semantic segmentation. Nevertheless, these networks are vulnerable to adversarial attacks that add imperceptible perturbations to the input image, leading to false predictions. This vulnerability is particularly dangerous in safety-critical applications like automated driving. While adversarial examples and defense strategies are well-researched in the context of image classification, there is comparatively less research focused on semantic segmentation. Recently, we have proposed an uncertainty-based method for detecting adversarial attacks on neural networks for semantic segmentation. We observed that uncertainty, as measured by the entropy of the output distribution, behaves differently on clean versus adversely perturbed images, and we utilize this property to differentiate between the two. In this extended version of our work, we conduct a detailed analysis of uncertainty-based detection of adversarial attacks including a diverse set of adversarial attacks and various state-of-the-art neural networks. Our numerical experiments show the effectiveness of the proposed uncertainty-based detection method, which is lightweight and operates as a post-processing step, i.e., no model modifications or knowledge of the adversarial example generation process are required.



## **33. Adversarial Prompt Tuning for Vision-Language Models**

cs.CV

ECCV 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2311.11261v3) [paper-pdf](http://arxiv.org/pdf/2311.11261v3)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.



## **34. Segment-Anything Models Achieve Zero-shot Robustness in Autonomous Driving**

cs.CV

Accepted to IAVVC 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09839v1) [paper-pdf](http://arxiv.org/pdf/2408.09839v1)

**Authors**: Jun Yan, Pengyu Wang, Danni Wang, Weiquan Huang, Daniel Watzenig, Huilin Yin

**Abstract**: Semantic segmentation is a significant perception task in autonomous driving. It suffers from the risks of adversarial examples. In the past few years, deep learning has gradually transitioned from convolutional neural network (CNN) models with a relatively small number of parameters to foundation models with a huge number of parameters. The segment-anything model (SAM) is a generalized image segmentation framework that is capable of handling various types of images and is able to recognize and segment arbitrary objects in an image without the need to train on a specific object. It is a unified model that can handle diverse downstream tasks, including semantic segmentation, object detection, and tracking. In the task of semantic segmentation for autonomous driving, it is significant to study the zero-shot adversarial robustness of SAM. Therefore, we deliver a systematic empirical study on the robustness of SAM without additional training. Based on the experimental results, the zero-shot adversarial robustness of the SAM under the black-box corruptions and white-box adversarial attacks is acceptable, even without the need for additional training. The finding of this study is insightful in that the gigantic model parameters and huge amounts of training data lead to the phenomenon of emergence, which builds a guarantee of adversarial robustness. SAM is a vision foundation model that can be regarded as an early prototype of an artificial general intelligence (AGI) pipeline. In such a pipeline, a unified model can handle diverse tasks. Therefore, this research not only inspects the impact of vision foundation models on safe autonomous driving but also provides a perspective on developing trustworthy AGI. The code is available at: https://github.com/momo1986/robust_sam_iv.



## **35. Patch of Invisibility: Naturalistic Physical Black-Box Adversarial Attacks on Object Detectors**

cs.CV

Accepted at MLCS @ ECML-PKDD 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2303.04238v5) [paper-pdf](http://arxiv.org/pdf/2303.04238v5)

**Authors**: Raz Lapid, Eylon Mizrahi, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called "white-box" attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a direct, black-box, gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. To our knowledge this is the first and only method that performs black-box physical attacks directly on object-detection models, which results with a model-agnostic attack. We show that our proposed method works both digitally and physically. We compared our approach against four different black-box attacks with different configurations. Our approach outperformed all other approaches that were tested in our experiments by a large margin.



## **36. Regularization for Adversarial Robust Learning**

cs.LG

51 pages, 5 figures

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09672v1) [paper-pdf](http://arxiv.org/pdf/2408.09672v1)

**Authors**: Jie Wang, Rui Gao, Yao Xie

**Abstract**: Despite the growing prevalence of artificial neural networks in real-world applications, their vulnerability to adversarial attacks remains to be a significant concern, which motivates us to investigate the robustness of machine learning models. While various heuristics aim to optimize the distributionally robust risk using the $\infty$-Wasserstein metric, such a notion of robustness frequently encounters computation intractability. To tackle the computational challenge, we develop a novel approach to adversarial training that integrates $\phi$-divergence regularization into the distributionally robust risk function. This regularization brings a notable improvement in computation compared with the original formulation. We develop stochastic gradient methods with biased oracles to solve this problem efficiently, achieving the near-optimal sample complexity. Moreover, we establish its regularization effects and demonstrate it is asymptotic equivalence to a regularized empirical risk minimization (ERM) framework, by considering various scaling regimes of the regularization parameter $\eta$ and robustness level $\rho$. These regimes yield gradient norm regularization, variance regularization, or a smoothed gradient norm regularization that interpolates between these extremes. We numerically validate our proposed method in supervised learning, reinforcement learning, and contextual learning and showcase its state-of-the-art performance against various adversarial attacks.



## **37. Symbiotic Game and Foundation Models for Cyber Deception Operations in Strategic Cyber Warfare**

cs.CR

40 pages, 7 figures, 2 tables

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2403.10570v2) [paper-pdf](http://arxiv.org/pdf/2403.10570v2)

**Authors**: Tao Li, Quanyan Zhu

**Abstract**: We are currently facing unprecedented cyber warfare with the rapid evolution of tactics, increasing asymmetry of intelligence, and the growing accessibility of hacking tools. In this landscape, cyber deception emerges as a critical component of our defense strategy against increasingly sophisticated attacks. This chapter aims to highlight the pivotal role of game-theoretic models and foundation models (FMs) in analyzing, designing, and implementing cyber deception tactics. Game models (GMs) serve as a foundational framework for modeling diverse adversarial interactions, allowing us to encapsulate both adversarial knowledge and domain-specific insights. Meanwhile, FMs serve as the building blocks for creating tailored machine learning models suited to given applications. By leveraging the synergy between GMs and FMs, we can advance proactive and automated cyber defense mechanisms by not only securing our networks against attacks but also enhancing their resilience against well-planned operations. This chapter discusses the games at the tactical, operational, and strategic levels of warfare, delves into the symbiotic relationship between these methodologies, and explores relevant applications where such a framework can make a substantial impact in cybersecurity. The chapter discusses the promising direction of the multi-agent neurosymbolic conjectural learning (MANSCOL), which allows the defender to predict adversarial behaviors, design adaptive defensive deception tactics, and synthesize knowledge for the operational level synthesis and adaptation. FMs serve as pivotal tools across various functions for MANSCOL, including reinforcement learning, knowledge assimilation, formation of conjectures, and contextual representation. This chapter concludes with a discussion of the challenges associated with FMs and their application in the domain of cybersecurity.



## **38. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

cs.CR

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2312.00029v3) [paper-pdf](http://arxiv.org/pdf/2312.00029v3)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. Such vulnerabilities can lead to LLMs being manipulated into generating hazardous content: from instructions for creating dangerous materials to inciting violence or endorsing unethical behaviors. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM acting as a guardian to the primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis reviews that by using Bergeron to complement models with existing alignment training, we can significantly improve the robustness and safety of multiple, commonly used commercial and open-source LLMs. Specifically, we found that models integrated with Bergeron are, on average, nearly seven times more resistant to attacks compared to models without such support.



## **39. Interpreting Global Perturbation Robustness of Image Models using Axiomatic Spectral Importance Decomposition**

cs.AI

Accepted by Transactions on Machine Learning Research (TMLR 2024)

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.01139v2) [paper-pdf](http://arxiv.org/pdf/2408.01139v2)

**Authors**: Róisín Luo, James McDermott, Colm O'Riordan

**Abstract**: Perturbation robustness evaluates the vulnerabilities of models, arising from a variety of perturbations, such as data corruptions and adversarial attacks. Understanding the mechanisms of perturbation robustness is critical for global interpretability. We present a model-agnostic, global mechanistic interpretability method to interpret the perturbation robustness of image models. This research is motivated by two key aspects. First, previous global interpretability works, in tandem with robustness benchmarks, e.g. mean corruption error (mCE), are not designed to directly interpret the mechanisms of perturbation robustness within image models. Second, we notice that the spectral signal-to-noise ratios (SNR) of perturbed natural images exponentially decay over the frequency. This power-law-like decay implies that: Low-frequency signals are generally more robust than high-frequency signals -- yet high classification accuracy can not be achieved by low-frequency signals alone. By applying Shapley value theory, our method axiomatically quantifies the predictive powers of robust features and non-robust features within an information theory framework. Our method, dubbed as \textbf{I-ASIDE} (\textbf{I}mage \textbf{A}xiomatic \textbf{S}pectral \textbf{I}mportance \textbf{D}ecomposition \textbf{E}xplanation), provides a unique insight into model robustness mechanisms. We conduct extensive experiments over a variety of vision models pre-trained on ImageNet to show that \textbf{I-ASIDE} can not only \textbf{measure} the perturbation robustness but also \textbf{provide interpretations} of its mechanisms.



## **40. WPN: An Unlearning Method Based on N-pair Contrastive Learning in Language Models**

cs.CL

ECAI 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09459v1) [paper-pdf](http://arxiv.org/pdf/2408.09459v1)

**Authors**: Guitao Chen, Yunshen Wang, Hongye Sun, Guang Chen

**Abstract**: Generative language models (LMs) offer numerous advantages but may produce inappropriate or harmful outputs due to the harmful knowledge acquired during pre-training. This knowledge often manifests as undesirable correspondences, such as "harmful prompts" leading to "harmful outputs," which our research aims to mitigate through unlearning techniques.However, existing unlearning methods based on gradient ascent can significantly impair the performance of LMs. To address this issue, we propose a novel approach called Weighted Positional N-pair (WPN) Learning, which leverages position-weighted mean pooling within an n-pair contrastive learning framework. WPN is designed to modify the output distribution of LMs by eliminating specific harmful outputs (e.g., replacing toxic responses with neutral ones), thereby transforming the model's behavior from "harmful prompt-harmful output" to "harmful prompt-harmless response".Experiments on OPT and GPT-NEO LMs show that WPN effectively reduces the proportion of harmful responses, achieving a harmless rate of up to 95.8\% while maintaining stable performance on nine common benchmarks (with less than 2\% degradation on average). Moreover, we provide empirical evidence to demonstrate WPN's ability to weaken the harmful correspondences in terms of generalizability and robustness, as evaluated on out-of-distribution test sets and under adversarial attacks.



## **41. XAI-Based Detection of Adversarial Attacks on Deepfake Detectors**

cs.CR

Accepted at TMLR 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2403.02955v2) [paper-pdf](http://arxiv.org/pdf/2403.02955v2)

**Authors**: Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, Yehudit Aperstein

**Abstract**: We introduce a novel methodology for identifying adversarial attacks on deepfake detectors using eXplainable Artificial Intelligence (XAI). In an era characterized by digital advancement, deepfakes have emerged as a potent tool, creating a demand for efficient detection systems. However, these systems are frequently targeted by adversarial attacks that inhibit their performance. We address this gap, developing a defensible deepfake detector by leveraging the power of XAI. The proposed methodology uses XAI to generate interpretability maps for a given method, providing explicit visualizations of decision-making factors within the AI models. We subsequently employ a pretrained feature extractor that processes both the input image and its corresponding XAI image. The feature embeddings extracted from this process are then used for training a simple yet effective classifier. Our approach contributes not only to the detection of deepfakes but also enhances the understanding of possible adversarial attacks, pinpointing potential vulnerabilities. Furthermore, this approach does not change the performance of the deepfake detector. The paper demonstrates promising results suggesting a potential pathway for future deepfake detection mechanisms. We believe this study will serve as a valuable contribution to the community, sparking much-needed discourse on safeguarding deepfake detectors.



## **42. Adversarial Attacked Teacher for Unsupervised Domain Adaptive Object Detection**

cs.CV

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09431v1) [paper-pdf](http://arxiv.org/pdf/2408.09431v1)

**Authors**: Kaiwen Wang, Yinzhe Shen, Martin Lauer

**Abstract**: Object detectors encounter challenges in handling domain shifts. Cutting-edge domain adaptive object detection methods use the teacher-student framework and domain adversarial learning to generate domain-invariant pseudo-labels for self-training. However, the pseudo-labels generated by the teacher model tend to be biased towards the majority class and often mistakenly include overconfident false positives and underconfident false negatives. We reveal that pseudo-labels vulnerable to adversarial attacks are more likely to be low-quality. To address this, we propose a simple yet effective framework named Adversarial Attacked Teacher (AAT) to improve the quality of pseudo-labels. Specifically, we apply adversarial attacks to the teacher model, prompting it to generate adversarial pseudo-labels to correct bias, suppress overconfidence, and encourage underconfident proposals. An adaptive pseudo-label regularization is introduced to emphasize the influence of pseudo-labels with high certainty and reduce the negative impacts of uncertain predictions. Moreover, robust minority objects verified by pseudo-label regularization are oversampled to minimize dataset imbalance without introducing false positives. Extensive experiments conducted on various datasets demonstrate that AAT achieves superior performance, reaching 52.6 mAP on Clipart1k, surpassing the previous state-of-the-art by 6.7%.



## **43. Rethinking Impersonation and Dodging Attacks on Face Recognition Systems**

cs.CV

Accepted to ACM MM 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2401.08903v4) [paper-pdf](http://arxiv.org/pdf/2401.08903v4)

**Authors**: Fengfan Zhou, Qianyu Zhou, Bangjie Yin, Hui Zheng, Xuequan Lu, Lizhuang Ma, Hefei Ling

**Abstract**: Face Recognition (FR) systems can be easily deceived by adversarial examples that manipulate benign face images through imperceptible perturbations. Adversarial attacks on FR encompass two types: impersonation (targeted) attacks and dodging (untargeted) attacks. Previous methods often achieve a successful impersonation attack on FR, however, it does not necessarily guarantee a successful dodging attack on FR in the black-box setting. In this paper, our key insight is that the generation of adversarial examples should perform both impersonation and dodging attacks simultaneously. To this end, we propose a novel attack method termed as Adversarial Pruning (Adv-Pruning), to fine-tune existing adversarial examples to enhance their dodging capabilities while preserving their impersonation capabilities. Adv-Pruning consists of Priming, Pruning, and Restoration stages. Concretely, we propose Adversarial Priority Quantification to measure the region-wise priority of original adversarial perturbations, identifying and releasing those with minimal impact on absolute model output variances. Then, Biased Gradient Adaptation is presented to adapt the adversarial examples to traverse the decision boundaries of both the attacker and victim by adding perturbations favoring dodging attacks on the vacated regions, preserving the prioritized features of the original perturbations while boosting dodging performance. As a result, we can maintain the impersonation capabilities of original adversarial examples while effectively enhancing dodging capabilities. Comprehensive experiments demonstrate the superiority of our method compared with state-of-the-art adversarial attack methods.



## **44. Malacopula: adversarial automatic speaker verification attacks using a neural-based generalised Hammerstein model**

eess.AS

Accepted at ASVspoof Workshop 2024

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09300v1) [paper-pdf](http://arxiv.org/pdf/2408.09300v1)

**Authors**: Massimiliano Todisco, Michele Panariello, Xin Wang, Héctor Delgado, Kong Aik Lee, Nicholas Evans

**Abstract**: We present Malacopula, a neural-based generalised Hammerstein model designed to introduce adversarial perturbations to spoofed speech utterances so that they better deceive automatic speaker verification (ASV) systems. Using non-linear processes to modify speech utterances, Malacopula enhances the effectiveness of spoofing attacks. The model comprises parallel branches of polynomial functions followed by linear time-invariant filters. The adversarial optimisation procedure acts to minimise the cosine distance between speaker embeddings extracted from spoofed and bona fide utterances. Experiments, performed using three recent ASV systems and the ASVspoof 2019 dataset, show that Malacopula increases vulnerabilities by a substantial margin. However, speech quality is reduced and attacks can be detected effectively under controlled conditions. The findings emphasise the need to identify new vulnerabilities and design defences to protect ASV systems from adversarial attacks in the wild.



## **45. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

cs.CV

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09181v1) [paper-pdf](http://arxiv.org/pdf/2408.09181v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications. However, conducting physical experiments is extremely time-consuming and labor-intensive. Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models. To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation. This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world. Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis. In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics. Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **46. Training Verifiably Robust Agents Using Set-Based Reinforcement Learning**

cs.LG

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09112v1) [paper-pdf](http://arxiv.org/pdf/2408.09112v1)

**Authors**: Manuel Wendl, Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Reinforcement learning often uses neural networks to solve complex control tasks. However, neural networks are sensitive to input perturbations, which makes their deployment in safety-critical environments challenging. This work lifts recent results from formally verifying neural networks against such disturbances to reinforcement learning in continuous state and action spaces using reachability analysis. While previous work mainly focuses on adversarial attacks for robust reinforcement learning, we train neural networks utilizing entire sets of perturbed inputs and maximize the worst-case reward. The obtained agents are verifiably more robust than agents obtained by related work, making them more applicable in safety-critical environments. This is demonstrated with an extensive empirical evaluation of four different benchmarks.



## **47. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09093v1) [paper-pdf](http://arxiv.org/pdf/2408.09093v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **48. HookChain: A new perspective for Bypassing EDR Solutions**

cs.CR

50 pages, 23 figures, HookChain, Bypass EDR, Evading EDR, IAT Hook,  Halo's Gate

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2404.16856v3) [paper-pdf](http://arxiv.org/pdf/2404.16856v3)

**Authors**: Helvio Carvalho Junior

**Abstract**: In the current digital security ecosystem, where threats evolve rapidly and with complexity, companies developing Endpoint Detection and Response (EDR) solutions are in constant search for innovations that not only keep up but also anticipate emerging attack vectors. In this context, this article introduces the HookChain, a look from another perspective at widely known techniques, which when combined, provide an additional layer of sophisticated evasion against traditional EDR systems. Through a precise combination of IAT Hooking techniques, dynamic SSN resolution, and indirect system calls, HookChain redirects the execution flow of Windows subsystems in a way that remains invisible to the vigilant eyes of EDRs that only act on Ntdll.dll, without requiring changes to the source code of the applications and malwares involved. This work not only challenges current conventions in cybersecurity but also sheds light on a promising path for future protection strategies, leveraging the understanding that continuous evolution is key to the effectiveness of digital security. By developing and exploring the HookChain technique, this study significantly contributes to the body of knowledge in endpoint security, stimulating the development of more robust and adaptive solutions that can effectively address the ever-changing dynamics of digital threats. This work aspires to inspire deep reflection and advancement in the research and development of security technologies that are always several steps ahead of adversaries.



## **49. Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models**

cs.AI

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08989v1) [paper-pdf](http://arxiv.org/pdf/2408.08989v1)

**Authors**: Qingyuan Zeng, Zhenzhong Wang, Yiu-ming Cheung, Min Jiang

**Abstract**: While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.



## **50. Stochastic Bandits Robust to Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08859v1) [paper-pdf](http://arxiv.org/pdf/2408.08859v1)

**Authors**: Xuchuang Wang, Jinhang Zuo, Xutong Liu, John C. S. Lui, Mohammad Hajiesmaili

**Abstract**: This paper investigates stochastic multi-armed bandit algorithms that are robust to adversarial attacks, where an attacker can first observe the learner's action and {then} alter their reward observation. We study two cases of this model, with or without the knowledge of an attack budget $C$, defined as an upper bound of the summation of the difference between the actual and altered rewards. For both cases, we devise two types of algorithms with regret bounds having additive or multiplicative $C$ dependence terms. For the known attack budget case, we prove our algorithms achieve the regret bound of ${O}((K/\Delta)\log T + KC)$ and $\tilde{O}(\sqrt{KTC})$ for the additive and multiplicative $C$ terms, respectively, where $K$ is the number of arms, $T$ is the time horizon, $\Delta$ is the gap between the expected rewards of the optimal arm and the second-best arm, and $\tilde{O}$ hides the logarithmic factors. For the unknown case, we prove our algorithms achieve the regret bound of $\tilde{O}(\sqrt{KT} + KC^2)$ and $\tilde{O}(KC\sqrt{T})$ for the additive and multiplicative $C$ terms, respectively. In addition to these upper bound results, we provide several lower bounds showing the tightness of our bounds and the optimality of our algorithms. These results delineate an intrinsic separation between the bandits with attacks and corruption models [Lykouris et al., 2018].



