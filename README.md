# Latest Adversarial Attack Papers
**update at 2024-10-08 09:47:04**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Suspiciousness of Adversarial Texts to Human**

cs.LG

Under review

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2410.04377v1) [paper-pdf](http://arxiv.org/pdf/2410.04377v1)

**Authors**: Shakila Mahjabin Tonni, Pedro Faustini, Mark Dras

**Abstract**: Adversarial examples pose a significant challenge to deep neural networks (DNNs) across both image and text domains, with the intent to degrade model performance through meticulously altered inputs. Adversarial texts, however, are distinct from adversarial images due to their requirement for semantic similarity and the discrete nature of the textual contents. This study delves into the concept of human suspiciousness, a quality distinct from the traditional focus on imperceptibility found in image-based adversarial examples. Unlike images, where adversarial changes are meant to be indistinguishable to the human eye, textual adversarial content must often remain undetected or non-suspicious to human readers, even when the text's purpose is to deceive NLP systems or bypass filters.   In this research, we expand the study of human suspiciousness by analyzing how individuals perceive adversarial texts. We gather and publish a novel dataset of Likert-scale human evaluations on the suspiciousness of adversarial sentences, crafted by four widely used adversarial attack methods and assess their correlation with the human ability to detect machine-generated alterations. Additionally, we develop a regression-based model to quantify suspiciousness and establish a baseline for future research in reducing the suspiciousness in adversarial text generation. We also demonstrate how the regressor-generated suspicious scores can be incorporated into adversarial generation methods to produce texts that are less likely to be perceived as computer-generated. We make our human suspiciousness annotated data and our code available.



## **2. Adversarial Suffixes May Be Features Too!**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.00451v2) [paper-pdf](http://arxiv.org/pdf/2410.00451v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including those triggered by adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets, i.e., even in the absence of harmful content. This highlights the critical risk posed by dominating benign features in the training data and calls for further research to reinforce LLM safety alignment. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.



## **3. Exploring Strengths and Weaknesses of Super-Resolution Attack in Deepfake Detection**

cs.CV

Trust What You learN (TWYN) Workshop at European Conference on  Computer Vision ECCV 2024

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04205v1) [paper-pdf](http://arxiv.org/pdf/2410.04205v1)

**Authors**: Davide Alessandro Coccomini, Roberto Caldelli, Fabrizio Falchi, Claudio Gennaro, Giuseppe Amato

**Abstract**: Image manipulation is rapidly evolving, allowing the creation of credible content that can be used to bend reality. Although the results of deepfake detectors are promising, deepfakes can be made even more complicated to detect through adversarial attacks. They aim to further manipulate the image to camouflage deepfakes' artifacts or to insert signals making the image appear pristine. In this paper, we further explore the potential of super-resolution attacks based on different super-resolution techniques and with different scales that can impact the performance of deepfake detectors with more or less intensity. We also evaluated the impact of the attack on more diverse datasets discovering that the super-resolution process is effective in hiding the artifacts introduced by deepfake generation models but fails in hiding the traces contained in fully synthetic images. Finally, we propose some changes to the detectors' training process to improve their robustness to this kind of attack.



## **4. Automated Progressive Red Teaming**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2407.03876v2) [paper-pdf](http://arxiv.org/pdf/2407.03876v2)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).



## **5. Panda or not Panda? Understanding Adversarial Attacks with Interactive Visualization**

cs.HC

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2311.13656v2) [paper-pdf](http://arxiv.org/pdf/2311.13656v2)

**Authors**: Yuzhe You, Jarvis Tse, Jian Zhao

**Abstract**: Adversarial machine learning (AML) studies attacks that can fool machine learning algorithms into generating incorrect outcomes as well as the defenses against worst-case attacks to strengthen model robustness. Specifically for image classification, it is challenging to understand adversarial attacks due to their use of subtle perturbations that are not human-interpretable, as well as the variability of attack impacts influenced by diverse methodologies, instance differences, and model architectures. Through a design study with AML learners and teachers, we introduce AdvEx, a multi-level interactive visualization system that comprehensively presents the properties and impacts of evasion attacks on different image classifiers for novice AML learners. We quantitatively and qualitatively assessed AdvEx in a two-part evaluation including user studies and expert interviews. Our results show that AdvEx is not only highly effective as a visualization tool for understanding AML mechanisms, but also provides an engaging and enjoyable learning experience, thus demonstrating its overall benefits for AML learners.



## **6. Adversarial Attacks and Robust Defenses in Speaker Embedding based Zero-Shot Text-to-Speech System**

eess.AS

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04017v1) [paper-pdf](http://arxiv.org/pdf/2410.04017v1)

**Authors**: Ze Li, Yao Shi, Yunfei Xu, Ming Li

**Abstract**: Speaker embedding based zero-shot Text-to-Speech (TTS) systems enable high-quality speech synthesis for unseen speakers using minimal data. However, these systems are vulnerable to adversarial attacks, where an attacker introduces imperceptible perturbations to the original speaker's audio waveform, leading to synthesized speech sounds like another person. This vulnerability poses significant security risks, including speaker identity spoofing and unauthorized voice manipulation. This paper investigates two primary defense strategies to address these threats: adversarial training and adversarial purification. Adversarial training enhances the model's robustness by integrating adversarial examples during the training process, thereby improving resistance to such attacks. Adversarial purification, on the other hand, employs diffusion probabilistic models to revert adversarially perturbed audio to its clean form. Experimental results demonstrate that these defense mechanisms can significantly reduce the impact of adversarial perturbations, enhancing the security and reliability of speaker embedding based zero-shot TTS systems in adversarial environments.



## **7. Impact of Regularization on Calibration and Robustness: from the Representation Space Perspective**

cs.CV

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.03999v1) [paper-pdf](http://arxiv.org/pdf/2410.03999v1)

**Authors**: Jonghyun Park, Juyeop Kim, Jong-Seok Lee

**Abstract**: Recent studies have shown that regularization techniques using soft labels, e.g., label smoothing, Mixup, and CutMix, not only enhance image classification accuracy but also improve model calibration and robustness against adversarial attacks. However, the underlying mechanisms of such improvements remain underexplored. In this paper, we offer a novel explanation from the perspective of the representation space (i.e., the space of the features obtained at the penultimate layer). Our investigation first reveals that the decision regions in the representation space form cone-like shapes around the origin after training regardless of the presence of regularization. However, applying regularization causes changes in the distribution of features (or representation vectors). The magnitudes of the representation vectors are reduced and subsequently the cosine similarities between the representation vectors and the class centers (minimal loss points for each class) become higher, which acts as a central mechanism inducing improved calibration and robustness. Our findings provide new insights into the characteristics of the high-dimensional representation space in relation to training and regularization using soft labels.



## **8. Adversarial Attacks on Data Attribution**

cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.05657v2) [paper-pdf](http://arxiv.org/pdf/2409.05657v2)

**Authors**: Xinhe Wang, Pingbang Hu, Junwei Deng, Jiaqi W. Ma

**Abstract**: Data attribution aims to quantify the contribution of individual training data points to the outputs of an AI model, which has been used to measure the value of training data and compensate data providers. Given the impact on financial decisions and compensation mechanisms, a critical question arises concerning the adversarial robustness of data attribution methods. However, there has been little to no systematic research addressing this issue. In this work, we aim to bridge this gap by detailing a threat model with clear assumptions about the adversary's goal and capabilities and proposing principled adversarial attack methods on data attribution. We present two methods, Shadow Attack and Outlier Attack, which generate manipulated datasets to inflate the compensation adversarially. The Shadow Attack leverages knowledge about the data distribution in the AI applications, and derives adversarial perturbations through "shadow training", a technique commonly used in membership inference attacks. In contrast, the Outlier Attack does not assume any knowledge about the data distribution and relies solely on black-box queries to the target model's predictions. It exploits an inductive bias present in many data attribution methods - outlier data points are more likely to be influential - and employs adversarial examples to generate manipulated datasets. Empirically, in image classification and text generation tasks, the Shadow Attack can inflate the data-attribution-based compensation by at least 200%, while the Outlier Attack achieves compensation inflation ranging from 185% to as much as 643%.



## **9. Detecting Machine-Generated Long-Form Content with Latent-Space Variables**

cs.CL

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03856v1) [paper-pdf](http://arxiv.org/pdf/2410.03856v1)

**Authors**: Yufei Tian, Zeyu Pan, Nanyun Peng

**Abstract**: The increasing capability of large language models (LLMs) to generate fluent long-form texts is presenting new challenges in distinguishing machine-generated outputs from human-written ones, which is crucial for ensuring authenticity and trustworthiness of expressions. Existing zero-shot detectors primarily focus on token-level distributions, which are vulnerable to real-world domain shifts, including different prompting and decoding strategies, and adversarial attacks. We propose a more robust method that incorporates abstract elements, such as event transitions, as key deciding factors to detect machine versus human texts by training a latent-space model on sequences of events or topics derived from human-written texts. In three different domains, machine-generated texts, which are originally inseparable from human texts on the token level, can be better distinguished with our latent-space model, leading to a 31% improvement over strong baselines such as DetectGPT. Our analysis further reveals that, unlike humans, modern LLMs like GPT-4 generate event triggers and their transitions differently, an inherent disparity that helps our method to robustly detect machine-generated texts.



## **10. Evaluation of Security of ML-based Watermarking: Copy and Removal Attacks**

cs.CV

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.18211v2) [paper-pdf](http://arxiv.org/pdf/2409.18211v2)

**Authors**: Vitaliy Kinakh, Brian Pulfer, Yury Belousov, Pierre Fernandez, Teddy Furon, Slava Voloshynovskiy

**Abstract**: The vast amounts of digital content captured from the real world or AI-generated media necessitate methods for copyright protection, traceability, or data provenance verification. Digital watermarking serves as a crucial approach to address these challenges. Its evolution spans three generations: handcrafted, autoencoder-based, and foundation model based methods. While the robustness of these systems is well-documented, the security against adversarial attacks remains underexplored. This paper evaluates the security of foundation models' latent space digital watermarking systems that utilize adversarial embedding techniques. A series of experiments investigate the security dimensions under copy and removal attacks, providing empirical insights into these systems' vulnerabilities. All experimental codes and results are available at https://github.com/vkinakh/ssl-watermarking-attacks .



## **11. RAFT: Realistic Attacks to Fool Text Detectors**

cs.CL

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03658v1) [paper-pdf](http://arxiv.org/pdf/2410.03658v1)

**Authors**: James Wang, Ran Li, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) have exhibited remarkable fluency across various tasks. However, their unethical applications, such as disseminating disinformation, have become a growing concern. Although recent works have proposed a number of LLM detection methods, their robustness and reliability remain unclear. In this paper, we present RAFT: a grammar error-free black-box attack against existing LLM detectors. In contrast to previous attacks for language models, our method exploits the transferability of LLM embeddings at the word-level while preserving the original text quality. We leverage an auxiliary embedding to greedily select candidate words to perturb against the target detector. Experiments reveal that our attack effectively compromises all detectors in the study across various domains by up to 99%, and are transferable across source models. Manual human evaluation studies show our attacks are realistic and indistinguishable from original human-written text. We also show that examples generated by RAFT can be used to train adversarially robust detectors. Our work shows that current LLM detectors are not adversarially robust, underscoring the urgent need for more resilient detection mechanisms.



## **12. Jailbreaking as a Reward Misspecification Problem**

cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.14393v3) [paper-pdf](http://arxiv.org/pdf/2406.14393v3)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.



## **13. Gradient-based Jailbreak Images for Multimodal Fusion Models**

cs.CR

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03489v1) [paper-pdf](http://arxiv.org/pdf/2410.03489v1)

**Authors**: Javier Rando, Hannah Korevaar, Erik Brinkman, Ivan Evtimov, Florian Tramèr

**Abstract**: Augmenting language models with image inputs may enable more effective jailbreak attacks through continuous optimization, unlike text inputs that require discrete optimization. However, new multimodal fusion models tokenize all input modalities using non-differentiable functions, which hinders straightforward attacks. In this work, we introduce the notion of a tokenizer shortcut that approximates tokenization with a continuous function and enables continuous optimization. We use tokenizer shortcuts to create the first end-to-end gradient image attacks against multimodal fusion models. We evaluate our attacks on Chameleon models and obtain jailbreak images that elicit harmful information for 72.5% of prompts. Jailbreak images outperform text jailbreaks optimized with the same objective and require 3x lower compute budget to optimize 50x more input tokens. Finally, we find that representation engineering defenses, like Circuit Breakers, trained only on text attacks can effectively transfer to adversarial image inputs.



## **14. Mitigating Adversarial Perturbations for Deep Reinforcement Learning via Vector Quantization**

cs.LG

8 pages, IROS 2024 (Code: https://github.com/tunglm2203/vq_robust_rl)

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03376v1) [paper-pdf](http://arxiv.org/pdf/2410.03376v1)

**Authors**: Tung M. Luu, Thanh Nguyen, Tee Joshua Tian Jin, Sungwoon Kim, Chang D. Yoo

**Abstract**: Recent studies reveal that well-performing reinforcement learning (RL) agents in training often lack resilience against adversarial perturbations during deployment. This highlights the importance of building a robust agent before deploying it in the real world. Most prior works focus on developing robust training-based procedures to tackle this problem, including enhancing the robustness of the deep neural network component itself or adversarially training the agent on strong attacks. In this work, we instead study an input transformation-based defense for RL. Specifically, we propose using a variant of vector quantization (VQ) as a transformation for input observations, which is then used to reduce the space of adversarial attacks during testing, resulting in the transformed observations being less affected by attacks. Our method is computationally efficient and seamlessly integrates with adversarial training, further enhancing the robustness of RL agents against adversarial attacks. Through extensive experiments in multiple environments, we demonstrate that using VQ as the input transformation effectively defends against adversarial attacks on the agent's observations.



## **15. The Vital Role of Gradient Clipping in Byzantine-Resilient Distributed Learning**

cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2405.14432v3) [paper-pdf](http://arxiv.org/pdf/2405.14432v3)

**Authors**: Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Ahmed Jellouli, Geovani Rizk, John Stephan

**Abstract**: Byzantine-resilient distributed machine learning seeks to achieve robust learning performance in the presence of misbehaving or adversarial workers. While state-of-the-art (SOTA) robust distributed gradient descent (Robust-DGD) methods were proven theoretically optimal, their empirical success has often relied on pre-aggregation gradient clipping. However, the currently considered static clipping strategy exhibits mixed results: improving robustness against some attacks while being ineffective or detrimental against others. We address this gap by proposing a principled adaptive clipping strategy, termed Adaptive Robust Clipping (ARC). We show that ARC consistently enhances the empirical robustness of SOTA Robust-DGD methods, while preserving the theoretical robustness guarantees. Our analysis shows that ARC provably improves the asymptotic convergence guarantee of Robust-DGD in the case when the model is well-initialized. We validate this theoretical insight through an exhaustive set of experiments on benchmark image classification tasks. We observe that the improvement induced by ARC is more pronounced in highly heterogeneous and adversarial settings.



## **16. Adversarial Challenges in Network Intrusion Detection Systems: Research Insights and Future Prospects**

cs.CR

35 pages

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.18736v2) [paper-pdf](http://arxiv.org/pdf/2409.18736v2)

**Authors**: Sabrine Ennaji, Fabio De Gaspari, Dorjan Hitaj, Alicia K Bidi, Luigi V. Mancini

**Abstract**: Machine learning has brought significant advances in cybersecurity, particularly in the development of Intrusion Detection Systems (IDS). These improvements are mainly attributed to the ability of machine learning algorithms to identify complex relationships between features and effectively generalize to unseen data. Deep neural networks, in particular, contributed to this progress by enabling the analysis of large amounts of training data, significantly enhancing detection performance. However, machine learning models remain vulnerable to adversarial attacks, where carefully crafted input data can mislead the model into making incorrect predictions. While adversarial threats in unstructured data, such as images and text, have been extensively studied, their impact on structured data like network traffic is less explored. This survey aims to address this gap by providing a comprehensive review of machine learning-based Network Intrusion Detection Systems (NIDS) and thoroughly analyzing their susceptibility to adversarial attacks. We critically examine existing research in NIDS, highlighting key trends, strengths, and limitations, while identifying areas that require further exploration. Additionally, we discuss emerging challenges in the field and offer insights for the development of more robust and resilient NIDS. In summary, this paper enhances the understanding of adversarial attacks and defenses in NIDS and guide future research in improving the robustness of machine learning models in cybersecurity applications.



## **17. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.02240v2) [paper-pdf](http://arxiv.org/pdf/2410.02240v2)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our code can be found at https://github.com/Pan-Zihao/SCA.



## **18. MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks**

eess.IV

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2401.09624v2) [paper-pdf](http://arxiv.org/pdf/2401.09624v2)

**Authors**: Giovanni Pasqualino, Luca Guarnera, Alessandro Ortis, Sebastiano Battiato

**Abstract**: The progress in generative models, particularly Generative Adversarial Networks (GANs), opened new possibilities for image generation but raised concerns about potential malicious uses, especially in sensitive areas like medical imaging. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing finely tuned perturbations that are imperceptible to the human eye. Specifically, the proposed approach involves the introduction of appropriate Gaussian noise to the input as a protective measure against various attacks. Our method aims to enhance tamper resistance, comparing favorably to existing techniques. Experimental results on a CT scan demonstrate MITS-GAN's superior performance, emphasizing its ability to generate tamper-resistant images with negligible artifacts. As image tampering in medical domains poses life-threatening risks, our proactive approach contributes to the responsible and ethical use of generative models. This work provides a foundation for future research in countering cyber threats in medical imaging. Models and codes are publicly available on https://iplab.dmi.unict.it/MITS-GAN-2024/.



## **19. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

math.OC

8 pages, 2 figures

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03218v1) [paper-pdf](http://arxiv.org/pdf/2410.03218v1)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are completely arbitrary. We show that when the probability of having an attack at a given time is less than 0.5, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.



## **20. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

cs.CL

Accepted to EMNLP 2024 Findings

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2401.07867v3) [paper-pdf](http://arxiv.org/pdf/2401.07867v3)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).



## **21. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

cs.LG

36 pages

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2407.11463v3) [paper-pdf](http://arxiv.org/pdf/2407.11463v3)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models by causing incorrect predictions through imperceptible perturbations to the input data. While these attacks have been extensively studied in unstructured data like images, applying them to tabular data, poses new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ from the image data. To account for this distinction, it is necessary to establish tailored imperceptibility criteria specific to tabular data. However, there is currently a lack of standardised metrics for assessing the imperceptibility of adversarial attacks on tabular data. To address this gap, we propose a set of key properties and corresponding metrics designed to comprehensively characterise imperceptible adversarial attacks on tabular data. These are: proximity to the original input, sparsity of altered features, deviation from the original data distribution, sensitivity in perturbing features with narrow distribution, immutability of certain features that should remain unchanged, feasibility of specific feature values that should not go beyond valid practical ranges, and feature interdependencies capturing complex relationships between data attributes. We evaluate the imperceptibility of five adversarial attacks, including both bounded attacks and unbounded attacks, on tabular data using the proposed imperceptibility metrics. The results reveal a trade-off between the imperceptibility and effectiveness of these attacks. The study also identifies limitations in current attack algorithms, offering insights that can guide future research in the area. The findings gained from this empirical analysis provide valuable direction for enhancing the design of adversarial attack algorithms, thereby advancing adversarial machine learning on tabular data.



## **22. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

cs.CR

29 pages

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.11295v3) [paper-pdf](http://arxiv.org/pdf/2409.11295v3)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have demonstrated remarkable potential in autonomously completing a wide range of tasks on real websites, significantly boosting human productivity. However, web tasks, such as booking flights, usually involve users' PII, which may be exposed to potential privacy risks if web agents accidentally interact with compromised websites, a scenario that remains largely unexplored in the literature. In this work, we narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a realistic threat model for attacks on the website, where we consider two adversarial targets: stealing users' specific PII or the entire user request. Then, we propose a novel attack method, termed Environmental Injection Attack (EIA). EIA injects malicious content designed to adapt well to environments where the agents operate and our work instantiates EIA specifically for privacy scenarios in web environments. We collect 177 action steps that involve diverse PII categories on realistic websites from the Mind2Web, and conduct experiments using one of the most capable generalist web agent frameworks to date. The results demonstrate that EIA achieves up to 70% ASR in stealing specific PII and 16% ASR for full user request. Additionally, by accessing the stealthiness and experimenting with a defensive system prompt, we indicate that EIA is hard to detect and mitigate. Notably, attacks that are not well adapted for a webpage can be detected via human inspection, leading to our discussion about the trade-off between security and autonomy. However, extra attackers' efforts can make EIA seamlessly adapted, rendering such supervision ineffective. Thus, we further discuss the defenses at the pre- and post-deployment stages of the websites without relying on human supervision and call for more advanced defense strategies.



## **23. Influence-based Attributions can be Manipulated**

cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2409.05208v4) [paper-pdf](http://arxiv.org/pdf/2409.05208v4)

**Authors**: Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri

**Abstract**: Influence Functions are a standard tool for attributing predictions to training data in a principled manner and are widely used in applications such as data valuation and fairness. In this work, we present realistic incentives to manipulate influence-based attributions and investigate whether these attributions can be \textit{systematically} tampered by an adversary. We show that this is indeed possible for logistic regression models trained on ResNet feature embeddings and standard tabular fairness datasets and provide efficient attacks with backward-friendly implementations. Our work raises questions on the reliability of influence-based attributions in adversarial circumstances. Code is available at : \url{https://github.com/infinite-pursuits/influence-based-attributions-can-be-manipulated}



## **24. Kick Bad Guys Out! Conditionally Activated Anomaly Detection in Federated Learning with Zero-Knowledge Proof Verification**

cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2310.04055v3) [paper-pdf](http://arxiv.org/pdf/2310.04055v3)

**Authors**: Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Qifan Zhang, Yuhang Yao, Salman Avestimehr, Chaoyang He

**Abstract**: Federated Learning (FL) systems are susceptible to adversarial attacks, where malicious clients submit poisoned models to disrupt the convergence or plant backdoors that cause the global model to misclassify some samples. Current defense methods are often impractical for real-world FL systems, as they either rely on unrealistic prior knowledge or cause accuracy loss even in the absence of attacks. Furthermore, these methods lack a protocol for verifying execution, leaving participants uncertain about the correct execution of the mechanism. To address these challenges, we propose a novel anomaly detection strategy that is designed for real-world FL systems. Our approach activates the defense only when potential attacks are detected, and enables the removal of malicious models without affecting the benign ones. Additionally, we incorporate zero-knowledge proofs to ensure the integrity of the proposed defense mechanism. Experimental results demonstrate the effectiveness of our approach in enhancing FL system security against a comprehensive set of adversarial attacks in various ML tasks.



## **25. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02916v1) [paper-pdf](http://arxiv.org/pdf/2410.02916v1)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.



## **26. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02890v1) [paper-pdf](http://arxiv.org/pdf/2410.02890v1)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.



## **27. Mitigating Dialogue Hallucination for Large Vision Language Models via Adversarial Instruction Tuning**

cs.CV

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2403.10492v3) [paper-pdf](http://arxiv.org/pdf/2403.10492v3)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LVLMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues powered by our novel Adversarial Question Generator (AQG), which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the zero-shot performance of state-of-the-art LVLMs drops significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT) that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive experiments show our proposed approach successfully reduces dialogue hallucination while maintaining performance.



## **28. Erasing Conceptual Knowledge from Language Models**

cs.CL

Project Page: https://elm.baulab.info

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02760v1) [paper-pdf](http://arxiv.org/pdf/2410.02760v1)

**Authors**: Rohit Gandikota, Sheridan Feucht, Samuel Marks, David Bau

**Abstract**: Concept erasure in language models has traditionally lacked a comprehensive evaluation framework, leading to incomplete assessments of effectiveness of erasure methods. We propose an evaluation paradigm centered on three critical criteria: innocence (complete knowledge removal), seamlessness (maintaining conditional fluent generation), and specificity (preserving unrelated task performance). Our evaluation metrics naturally motivate the development of Erasure of Language Memory (ELM), a new method designed to address all three dimensions. ELM employs targeted low-rank updates to alter output distributions for erased concepts while preserving overall model capabilities including fluency when prompted for an erased concept. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative analysis shows that ELM achieves superior performance across our proposed metrics, including near-random scores on erased topic assessments, generation fluency, maintained accuracy on unrelated benchmarks, and robustness under adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info



## **29. Does Refusal Training in LLMs Generalize to the Past Tense?**

cs.CL

Update in v3: o1-mini and o1-preview results (on top of GPT-4o and  Claude 3.5 Sonnet added in v2). We provide code and jailbreak artifacts at  https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2407.11969v3) [paper-pdf](http://arxiv.org/pdf/2407.11969v3)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.



## **30. Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems**

cs.MA

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02506v1) [paper-pdf](http://arxiv.org/pdf/2410.02506v1)

**Authors**: Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, Tianlong Chen

**Abstract**: Recent advancements in large language model (LLM)-powered agents have shown that collective intelligence can significantly outperform individual capabilities, largely attributed to the meticulously designed inter-agent communication topologies. Though impressive in performance, existing multi-agent pipelines inherently introduce substantial token overhead, as well as increased economic costs, which pose challenges for their large-scale deployments. In response to this challenge, we propose an economical, simple, and robust multi-agent communication framework, termed $\texttt{AgentPrune}$, which can seamlessly integrate into mainstream multi-agent systems and prunes redundant or even malicious communication messages. Technically, $\texttt{AgentPrune}$ is the first to identify and formally define the \textit{communication redundancy} issue present in current LLM-based multi-agent pipelines, and efficiently performs one-shot pruning on the spatial-temporal message-passing graph, yielding a token-economic and high-performing communication topology. Extensive experiments across six benchmarks demonstrate that $\texttt{AgentPrune}$ \textbf{(I)} achieves comparable results as state-of-the-art topologies at merely $\$5.6$ cost compared to their $\$43.7$, \textbf{(II)} integrates seamlessly into existing multi-agent frameworks with $28.1\%\sim72.8\%\downarrow$ token reduction, and \textbf{(III)} successfully defend against two types of agent-based adversarial attacks with $3.5\%\sim10.8\%\uparrow$ performance boost.



## **31. Tradeoffs Between Alignment and Helpfulness in Language Models with Representation Engineering**

cs.CL

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2401.16332v4) [paper-pdf](http://arxiv.org/pdf/2401.16332v4)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.



## **32. Demonstration Attack against In-Context Learning for Code Intelligence**

cs.CR

17 pages, 5 figures

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02841v1) [paper-pdf](http://arxiv.org/pdf/2410.02841v1)

**Authors**: Yifei Ge, Weisong Sun, Yihang Lou, Chunrong Fang, Yiran Zhang, Yiming Li, Xiaofang Zhang, Yang Liu, Zhihong Zhao, Zhenyu Chen

**Abstract**: Recent advancements in large language models (LLMs) have revolutionized code intelligence by improving programming productivity and alleviating challenges faced by software developers. To further improve the performance of LLMs on specific code intelligence tasks and reduce training costs, researchers reveal a new capability of LLMs: in-context learning (ICL). ICL allows LLMs to learn from a few demonstrations within a specific context, achieving impressive results without parameter updating. However, the rise of ICL introduces new security vulnerabilities in the code intelligence field. In this paper, we explore a novel security scenario based on the ICL paradigm, where attackers act as third-party ICL agencies and provide users with bad ICL content to mislead LLMs outputs in code intelligence tasks. Our study demonstrates the feasibility and risks of such a scenario, revealing how attackers can leverage malicious demonstrations to construct bad ICL content and induce LLMs to produce incorrect outputs, posing significant threats to system security. We propose a novel method to construct bad ICL content called DICE, which is composed of two stages: Demonstration Selection and Bad ICL Construction, constructing targeted bad ICL content based on the user query and transferable across different query inputs. Ultimately, our findings emphasize the critical importance of securing ICL mechanisms to protect code intelligence systems from adversarial manipulation.



## **33. Fake It Until You Break It: On the Adversarial Robustness of AI-generated Image Detectors**

cs.CV

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.01574v2) [paper-pdf](http://arxiv.org/pdf/2410.01574v2)

**Authors**: Sina Mavali, Jonas Ricker, David Pape, Yash Sharma, Asja Fischer, Lea Schönherr

**Abstract**: While generative AI (GenAI) offers countless possibilities for creative and productive tasks, artificially generated media can be misused for fraud, manipulation, scams, misinformation campaigns, and more. To mitigate the risks associated with maliciously generated media, forensic classifiers are employed to identify AI-generated content. However, current forensic classifiers are often not evaluated in practically relevant scenarios, such as the presence of an attacker or when real-world artifacts like social media degradations affect images. In this paper, we evaluate state-of-the-art AI-generated image (AIGI) detectors under different attack scenarios. We demonstrate that forensic classifiers can be effectively attacked in realistic settings, even when the attacker does not have access to the target model and post-processing occurs after the adversarial examples are created, which is standard on social media platforms. These attacks can significantly reduce detection accuracy to the extent that the risks of relying on detectors outweigh their benefits. Finally, we propose a simple defense mechanism to make CLIP-based detectors, which are currently the best-performing detectors, robust against these attacks.



## **34. MOREL: Enhancing Adversarial Robustness through Multi-Objective Representation Learning**

cs.LG

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.01697v2) [paper-pdf](http://arxiv.org/pdf/2410.01697v2)

**Authors**: Sedjro Salomon Hotegni, Sebastian Peitz

**Abstract**: Extensive research has shown that deep neural networks (DNNs) are vulnerable to slight adversarial perturbations$-$small changes to the input data that appear insignificant but cause the model to produce drastically different outputs. In addition to augmenting training data with adversarial examples generated from a specific attack method, most of the current defense strategies necessitate modifying the original model architecture components to improve robustness or performing test-time data purification to handle adversarial attacks. In this work, we demonstrate that strong feature representation learning during training can significantly enhance the original model's robustness. We propose MOREL, a multi-objective feature representation learning approach, encouraging classification models to produce similar features for inputs within the same class, despite perturbations. Our training method involves an embedding space where cosine similarity loss and multi-positive contrastive loss are used to align natural and adversarial features from the model encoder and ensure tight clustering. Concurrently, the classifier is motivated to achieve accurate predictions. Through extensive experiments, we demonstrate that our approach significantly enhances the robustness of DNNs against white-box and black-box adversarial attacks, outperforming other methods that similarly require no architectural changes or test-time data purification. Our code is available at https://github.com/salomonhotegni/MOREL



## **35. AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples**

cs.LG

https://attackbench.github.io

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2404.19460v2) [paper-pdf](http://arxiv.org/pdf/2404.19460v2)

**Authors**: Antonio Emanuele Cinà, Jérôme Rony, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed, Fabio Roli

**Abstract**: Adversarial examples are typically optimized with gradient-based attacks. While novel attacks are continuously proposed, each is shown to outperform its predecessors using different experimental setups, hyperparameter settings, and number of forward and backward calls to the target models. This provides overly-optimistic and even biased evaluations that may unfairly favor one particular attack over the others. In this work, we aim to overcome these limitations by proposing AttackBench, i.e., the first evaluation framework that enables a fair comparison among different attacks. To this end, we first propose a categorization of gradient-based attacks, identifying their main components and differences. We then introduce our framework, which evaluates their effectiveness and efficiency. We measure these characteristics by (i) defining an optimality metric that quantifies how close an attack is to the optimal solution, and (ii) limiting the number of forward and backward queries to the model, such that all attacks are compared within a given maximum query budget. Our extensive experimental analysis compares more than $100$ attack implementations with a total of over $800$ different configurations against CIFAR-10 and ImageNet models, highlighting that only very few attacks outperform all the competing approaches. Within this analysis, we shed light on several implementation issues that prevent many attacks from finding better solutions or running at all. We release AttackBench as a publicly-available benchmark, aiming to continuously update it to include and evaluate novel gradient-based attacks for optimizing adversarial examples.



## **36. The Role of piracy in quantum proofs**

quant-ph

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02228v1) [paper-pdf](http://arxiv.org/pdf/2410.02228v1)

**Authors**: Anne Broadbent, Alex B. Grilo, Supartha Podder, Jamie Sikora

**Abstract**: A well-known feature of quantum information is that it cannot, in general, be cloned. Recently, a number of quantum-enabled information-processing tasks have demonstrated various forms of uncloneability; among these forms, piracy is an adversarial model that gives maximal power to the adversary, in controlling both a cloning-type attack, as well as the evaluation/verification stage. Here, we initiate the study of anti-piracy proof systems, which are proof systems that inherently prevent piracy attacks. We define anti-piracy proof systems, demonstrate such a proof system for an oracle problem, and also describe a candidate anti-piracy proof system for NP. We also study quantum proof systems that are cloneable and settle the famous QMA vs. QMA(2) debate in this setting. Lastly, we discuss how one can approach the QMA vs. QCMA question, by studying its cloneable variants.



## **37. Semantic-Aware Adversarial Training for Reliable Deep Hashing Retrieval**

cs.CV

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2310.14637v2) [paper-pdf](http://arxiv.org/pdf/2310.14637v2)

**Authors**: Xu Yuan, Zheng Zhang, Xunguang Wang, Lin Wu

**Abstract**: Deep hashing has been intensively studied and successfully applied in large-scale image retrieval systems due to its efficiency and effectiveness. Recent studies have recognized that the existence of adversarial examples poses a security threat to deep hashing models, that is, adversarial vulnerability. Notably, it is challenging to efficiently distill reliable semantic representatives for deep hashing to guide adversarial learning, and thereby it hinders the enhancement of adversarial robustness of deep hashing-based retrieval models. Moreover, current researches on adversarial training for deep hashing are hard to be formalized into a unified minimax structure. In this paper, we explore Semantic-Aware Adversarial Training (SAAT) for improving the adversarial robustness of deep hashing models. Specifically, we conceive a discriminative mainstay features learning (DMFL) scheme to construct semantic representatives for guiding adversarial learning in deep hashing. Particularly, our DMFL with the strict theoretical guarantee is adaptively optimized in a discriminative learning manner, where both discriminative and semantic properties are jointly considered. Moreover, adversarial examples are fabricated by maximizing the Hamming distance between the hash codes of adversarial samples and mainstay features, the efficacy of which is validated in the adversarial attack trials. Further, we, for the first time, formulate the formalized adversarial training of deep hashing into a unified minimax optimization under the guidance of the generated mainstay codes. Extensive experiments on benchmark datasets show superb attack performance against the state-of-the-art algorithms, meanwhile, the proposed adversarial training can effectively eliminate adversarial perturbations for trustworthy deep hashing-based retrieval. Our code is available at https://github.com/xandery-geek/SAAT.



## **38. Securing Cloud File Systems with Trusted Execution**

cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2305.18639v3) [paper-pdf](http://arxiv.org/pdf/2305.18639v3)

**Authors**: Quinn Burke, Yohan Beugin, Blaine Hoak, Rachel King, Eric Pauley, Ryan Sheatsley, Mingli Yu, Ting He, Thomas La Porta, Patrick McDaniel

**Abstract**: Cloud file systems offer organizations a scalable and reliable file storage solution. However, cloud file systems have become prime targets for adversaries, and traditional designs are not equipped to protect organizations against the myriad of attacks that may be initiated by a malicious cloud provider, co-tenant, or end-client. Recently proposed designs leveraging cryptographic techniques and trusted execution environments (TEEs) still force organizations to make undesirable trade-offs, consequently leading to either security, functional, or performance limitations. In this paper, we introduce BFS, a cloud file system that leverages the security capabilities provided by TEEs to bootstrap new security protocols that deliver strong security guarantees, high-performance, and a transparent POSIX-like interface to clients. BFS delivers stronger security guarantees and up to a 2.5X speedup over a state-of-the-art secure file system. Moreover, compared to the industry standard NFS, BFS achieves up to 2.2X speedups across micro-benchmarks and incurs <1X overhead for most macro-benchmark workloads. BFS demonstrates a holistic cloud file system design that does not sacrifice an organizations' security yet can embrace all of the functional and performance advantages of outsourcing.



## **39. Impact of White-Box Adversarial Attacks on Convolutional Neural Networks**

cs.CR

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.02043v1) [paper-pdf](http://arxiv.org/pdf/2410.02043v1)

**Authors**: Rakesh Podder, Sudipto Ghosh

**Abstract**: Autonomous vehicle navigation and healthcare diagnostics are among the many fields where the reliability and security of machine learning models for image data are critical. We conduct a comprehensive investigation into the susceptibility of Convolutional Neural Networks (CNNs), which are widely used for image data, to white-box adversarial attacks. We investigate the effects of various sophisticated attacks -- Fast Gradient Sign Method, Basic Iterative Method, Jacobian-based Saliency Map Attack, Carlini & Wagner, Projected Gradient Descent, and DeepFool -- on CNN performance metrics, (e.g., loss, accuracy), the differential efficacy of adversarial techniques in increasing error rates, the relationship between perceived image quality metrics (e.g., ERGAS, PSNR, SSIM, and SAM) and classification performance, and the comparative effectiveness of iterative versus single-step attacks. Using the MNIST, CIFAR-10, CIFAR-100, and Fashio_MNIST datasets, we explore the effect of different attacks on the CNNs performance metrics by varying the hyperparameters of CNNs. Our study provides insights into the robustness of CNNs against adversarial threats, pinpoints vulnerabilities, and underscores the urgent need for developing robust defense mechanisms to protect CNNs and ensuring their trustworthy deployment in real-world scenarios.



## **40. EAB-FL: Exacerbating Algorithmic Bias through Model Poisoning Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.02042v1) [paper-pdf](http://arxiv.org/pdf/2410.02042v1)

**Authors**: Syed Irfan Ali Meerza, Jian Liu

**Abstract**: Federated Learning (FL) is a technique that allows multiple parties to train a shared model collaboratively without disclosing their private data. It has become increasingly popular due to its distinct privacy advantages. However, FL models can suffer from biases against certain demographic groups (e.g., racial and gender groups) due to the heterogeneity of data and party selection. Researchers have proposed various strategies for characterizing the group fairness of FL algorithms to address this issue. However, the effectiveness of these strategies in the face of deliberate adversarial attacks has not been fully explored. Although existing studies have revealed various threats (e.g., model poisoning attacks) against FL systems caused by malicious participants, their primary aim is to decrease model accuracy, while the potential of leveraging poisonous model updates to exacerbate model unfairness remains unexplored. In this paper, we propose a new type of model poisoning attack, EAB-FL, with a focus on exacerbating group unfairness while maintaining a good level of model utility. Extensive experiments on three datasets demonstrate the effectiveness and efficiency of our attack, even with state-of-the-art fairness optimization algorithms and secure aggregation rules employed.



## **41. CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks**

cs.CV

ECCV 2024

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2407.10179v3) [paper-pdf](http://arxiv.org/pdf/2407.10179v3)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.



## **42. Social Media Authentication and Combating Deepfakes using Semi-fragile Invisible Image Watermarking**

cs.CV

ACM Transactions (Digital Threats: Research and Practice)

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01906v1) [paper-pdf](http://arxiv.org/pdf/2410.01906v1)

**Authors**: Aakash Varma Nadimpalli, Ajita Rattani

**Abstract**: With the significant advances in deep generative models for image and video synthesis, Deepfakes and manipulated media have raised severe societal concerns. Conventional machine learning classifiers for deepfake detection often fail to cope with evolving deepfake generation technology and are susceptible to adversarial attacks. Alternatively, invisible image watermarking is being researched as a proactive defense technique that allows media authentication by verifying an invisible secret message embedded in the image pixels. A handful of invisible image watermarking techniques introduced for media authentication have proven vulnerable to basic image processing operations and watermark removal attacks. In response, we have proposed a semi-fragile image watermarking technique that embeds an invisible secret message into real images for media authentication. Our proposed watermarking framework is designed to be fragile to facial manipulations or tampering while being robust to benign image-processing operations and watermark removal attacks. This is facilitated through a unique architecture of our proposed technique consisting of critic and adversarial networks that enforce high image quality and resiliency to watermark removal efforts, respectively, along with the backbone encoder-decoder and the discriminator networks. Thorough experimental investigations on SOTA facial Deepfake datasets demonstrate that our proposed model can embed a $64$-bit secret as an imperceptible image watermark that can be recovered with a high-bit recovery accuracy when benign image processing operations are applied while being non-recoverable when unseen Deepfake manipulations are applied. In addition, our proposed watermarking technique demonstrates high resilience to several white-box and black-box watermark removal attacks. Thus, obtaining state-of-the-art performance.



## **43. MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification**

cs.CV

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2406.05927v2) [paper-pdf](http://arxiv.org/pdf/2406.05927v2)

**Authors**: Sajjad Amini, Mohammadreza Teymoorianfard, Shiqing Ma, Amir Houmansadr

**Abstract**: We present a simple yet effective method to improve the robustness of both Convolutional and attention-based Neural Networks against adversarial examples by post-processing an adversarially trained model. Our technique, MeanSparse, cascades the activation functions of a trained model with novel operators that sparsify mean-centered feature vectors. This is equivalent to reducing feature variations around the mean, and we show that such reduced variations merely affect the model's utility, yet they strongly attenuate the adversarial perturbations and decrease the attacker's success rate. Our experiments show that, when applied to the top models in the RobustBench leaderboard, MeanSparse achieves a new robustness record of 75.28% (from 73.71%), 44.78% (from 42.67%) and 62.12% (from 59.56%) on CIFAR-10, CIFAR-100 and ImageNet, respectively, in terms of AutoAttack accuracy. Code is available at https://github.com/SPIN-UMass/MeanSparse



## **44. KeyVisor -- A Lightweight ISA Extension for Protected Key Handles with CPU-enforced Usage Policies**

cs.CR

preprint

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01777v1) [paper-pdf](http://arxiv.org/pdf/2410.01777v1)

**Authors**: Fabian Schwarz, Jan Philipp Thoma, Christian Rossow, Tim Güneysu

**Abstract**: The confidentiality of cryptographic keys is essential for the security of protection schemes used for communication, file encryption, and outsourced computation. Beyond cryptanalytic attacks, adversaries can steal keys from memory via software exploits or side channels, enabling them to, e.g., tamper with secrets or impersonate key owners. Therefore, existing defenses protect keys in dedicated devices or isolated memory, or store them only in encrypted form. However, these designs often provide unfavorable tradeoffs, sacrificing performance, fine-grained access control, or deployability.   In this paper, we present KeyVisor, a lightweight ISA extension that securely offloads the handling of cryptographic keys to the CPU. KeyVisor provides CPU instructions that enable applications to request protected key handles and perform AEAD cipher operations on them. The underlying keys are accessible only by KeyVisor, and thus never leak to memory. KeyVisor's direct CPU integration enables fast crypto operations and hardware-enforced key usage restrictions, e.g., keys usable only for de-/encryption, with a limited lifetime, or with a process binding. Furthermore, privileged software, e.g., the monitor firmware of TEEs, can revoke keys or bind them to a specific process/TEE. We implement KeyVisor for RISC-V based on Rocket Chip, evaluate its performance, and demonstrate real-world use cases, including key-value databases, automotive feature licensing, and a read-only network middlebox.



## **45. Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective**

cs.CV

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2404.14309v2) [paper-pdf](http://arxiv.org/pdf/2404.14309v2)

**Authors**: Yiming Liu, Kezhao Liu, Yao Xiao, Ziyi Dong, Xiaogang Xu, Pengxu Wei, Liang Lin

**Abstract**: Diffusion-Based Purification (DBP) has emerged as an effective defense mechanism against adversarial attacks. The efficacy of DBP has been attributed to the forward diffusion process, which narrows the distribution gap between clean and adversarial images through the addition of Gaussian noise. Although this explanation has some theoretical support, the significance of its contribution to robustness remains unclear. In this paper, we argue that the inherent stochasticity in the DBP process is the primary driver of its robustness. To explore this, we introduce a novel Deterministic White-Box (DW-box) evaluation protocol to assess robustness in the absence of stochasticity and to analyze the attack trajectories and loss landscapes. Our findings suggest that DBP models primarily leverage stochasticity to evade effective attack directions, and their ability to purify adversarial perturbations can be weak. To further enhance the robustness of DBP models, we introduce Adversarial Denoising Diffusion Training (ADDT), which incorporates classifier-guided adversarial perturbations into diffusion training, thereby strengthening the DBP models' ability to purify adversarial perturbations. Additionally, we propose Rank-Based Gaussian Mapping (RBGM) to make perturbations more compatible with diffusion models. Experimental results validate the effectiveness of ADDT. In conclusion, our study suggests that future research on DBP can benefit from the perspective of decoupling the stochasticity-based and purification-based robustness.



## **46. On Scaling LT-Coded Blockchains in Heterogeneous Networks and their Vulnerabilities to DoS Threats**

cs.IT

Extended version of the results presented at IEEE ICC 2024

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2402.05620v2) [paper-pdf](http://arxiv.org/pdf/2402.05620v2)

**Authors**: Harikrishnan K., J. Harshan, Anwitaman Datta

**Abstract**: Coded blockchains have acquired prominence as a promising solution to reduce storage costs and facilitate scalability. Within this class, Luby Transform (LT) coded blockchains are an appealing choice for scalability owing to the availability of a wide range of low-complexity decoders. In the first part of this work, we identify that traditional LT decoders like Belief Propagation and On-the-Fly Gaussian Elimination may not be optimal for heterogeneous networks with nodes that have varying computational and download capabilities. To address this, we introduce a family of hybrid decoders for LT codes and propose optimal operating regimes for them to recover the blockchain at the lowest decoding cost. While LT coded blockchain architecture has been studied from the aspects of storage savings and scalability, not much is known in terms of its security vulnerabilities. Pointing at this research gap, in the second part, we present novel denial-of-service threats on LT coded blockchains that target nodes with specific decoding capabilities, preventing them from joining the network. Our proposed threats are non-oblivious in nature, wherein adversaries gain access to the archived blocks, and choose to execute their attack on a subset of them based on underlying coding scheme. We show that our optimized threats can achieve the same level of damage as that of blind attacks, however, with limited amount of resources. Overall, this is the first work of its kind that opens up new questions on designing coded blockchains to jointly provide storage savings, scalability and also resilience to optimized threats.



## **47. On Using Certified Training towards Empirical Robustness**

cs.LG

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01617v1) [paper-pdf](http://arxiv.org/pdf/2410.01617v1)

**Authors**: Alessandro De Palma, Serge Durand, Zakaria Chihani, François Terrier, Caterina Urban

**Abstract**: Adversarial training is arguably the most popular way to provide empirical robustness against specific adversarial examples. While variants based on multi-step attacks incur significant computational overhead, single-step variants are vulnerable to a failure mode known as catastrophic overfitting, which hinders their practical utility for large perturbations. A parallel line of work, certified training, has focused on producing networks amenable to formal guarantees of robustness against any possible attack. However, the wide gap between the best-performing empirical and certified defenses has severely limited the applicability of the latter. Inspired by recent developments in certified training, which rely on a combination of adversarial attacks with network over-approximations, and by the connections between local linearity and catastrophic overfitting, we present experimental evidence on the practical utility and limitations of using certified training towards empirical robustness. We show that, when tuned for the purpose, a recent certified training algorithm can prevent catastrophic overfitting on single-step attacks, and that it can bridge the gap to multi-step baselines under appropriate experimental settings. Finally, we present a novel regularizer for network over-approximations that can achieve similar effects while markedly reducing runtime.



## **48. Automated Red Teaming with GOAT: the Generative Offensive Agent Tester**

cs.LG

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01606v1) [paper-pdf](http://arxiv.org/pdf/2410.01606v1)

**Authors**: Maya Pavlova, Erik Brinkman, Krithika Iyer, Vitor Albiero, Joanna Bitton, Hailey Nguyen, Joe Li, Cristian Canton Ferrer, Ivan Evtimov, Aaron Grattafiori

**Abstract**: Red teaming assesses how large language models (LLMs) can produce content that violates norms, policies, and rules set during their safety training. However, most existing automated methods in the literature are not representative of the way humans tend to interact with AI models. Common users of AI models may not have advanced knowledge of adversarial machine learning methods or access to model internals, and they do not spend a lot of time crafting a single highly effective adversarial prompt. Instead, they are likely to make use of techniques commonly shared online and exploit the multiturn conversational nature of LLMs. While manual testing addresses this gap, it is an inefficient and often expensive process. To address these limitations, we introduce the Generative Offensive Agent Tester (GOAT), an automated agentic red teaming system that simulates plain language adversarial conversations while leveraging multiple adversarial prompting techniques to identify vulnerabilities in LLMs. We instantiate GOAT with 7 red teaming attacks by prompting a general-purpose model in a way that encourages reasoning through the choices of methods available, the current target model's response, and the next steps. Our approach is designed to be extensible and efficient, allowing human testers to focus on exploring new areas of risk while automation covers the scaled adversarial stress-testing of known risk territory. We present the design and evaluation of GOAT, demonstrating its effectiveness in identifying vulnerabilities in state-of-the-art LLMs, with an ASR@10 of 97% against Llama 3.1 and 88% against GPT-4 on the JailbreakBench dataset.



## **49. $σ$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples**

cs.LG

Code available at  https://github.com/Cinofix/sigma-zero-adversarial-attack

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2402.01879v2) [paper-pdf](http://arxiv.org/pdf/2402.01879v2)

**Authors**: Antonio Emanuele Cinà, Francesco Villani, Maura Pintor, Lea Schönherr, Battista Biggio, Marcello Pelillo

**Abstract**: Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages a differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving robust and non-robust models, show that $\sigma$-zero finds minimum $\ell_0$-norm adversarial examples without requiring any time-consuming hyperparameter tuning, and that it outperforms all competing sparse attacks in terms of success rate, perturbation size, and efficiency.



## **50. Signal Adversarial Examples Generation for Signal Detection Network via White-Box Attack**

cs.CV

18 pages, 6 figures, submitted to Mobile Networks and Applications

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01393v1) [paper-pdf](http://arxiv.org/pdf/2410.01393v1)

**Authors**: Dongyang Li, Linyuan Wang, Guangwei Xiong, Bin Yan, Dekui Ma, Jinxian Peng

**Abstract**: With the development and application of deep learning in signal detection tasks, the vulnerability of neural networks to adversarial attacks has also become a security threat to signal detection networks. This paper defines a signal adversarial examples generation model for signal detection network from the perspective of adding perturbations to the signal. The model uses the inequality relationship of L2-norm between time domain and time-frequency domain to constrain the energy of signal perturbations. Building upon this model, we propose a method for generating signal adversarial examples utilizing gradient-based attacks and Short-Time Fourier Transform. The experimental results show that under the constraint of signal perturbation energy ratio less than 3%, our adversarial attack resulted in a 28.1% reduction in the mean Average Precision (mAP), a 24.7% reduction in recall, and a 30.4% reduction in precision of the signal detection network. Compared to random noise perturbation of equivalent intensity, our adversarial attack demonstrates a significant attack effect.



