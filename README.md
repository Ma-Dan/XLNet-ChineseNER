# XLNet Chinese NER

基于Bi-LSTM + CRF 的中文机构名、人名、地名识别，MSRA NER语料，BIO标注

CMU XLNet

参考资料: 

https://github.com/yanwii/ChineseNER

https://github.com/macanv/BERT-BiLSTM-CRF-NER
      
https://github.com/zihangdai/xlnet
      
https://github.com/ymcui/Chinese-PreTrained-XLNet

# 下载xlnet中文预训练模型
    参考 https://github.com/ymcui/Chinese-PreTrained-XLNet

    放到根目录 **chinese_xlnet_base_L-12_H-768_A-12** 下

# 用法

    # 训练
    python3 model.py --entry train

    # 预测
    python3 model.py --entry predict

# 介绍

### xlnet 模型的加载和使用

    def xlnet_layer(self):
        # 加载bert配置文件
        xlnet_config = xlnet.XLNetConfig(json_path = FLAGS.xlnet_config)
        run_config = xlnet.create_run_config(self.is_training, True, FLAGS)

        # 创建bert模型　
        xlnet_model = xlnet.XLNetModel(
            xlnet_config = xlnet_config,
            run_config = run_config,
            input_ids = self.input_ids,
            seg_ids = self.segment_ids,
            input_mask = self.input_mask)

        # 加载词向量
        self.embedded = xlnet_model.get_sequence_output()
        self.model_inputs = tf.nn.dropout(
            self.embedded, self.dropout
        )

### xlnet 优化器

    self.train_op, self.learning_rate, _ = model_utils.get_train_op(FLAGS, self.loss)
