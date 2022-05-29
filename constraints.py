from training.supervised.constraints import * # DL2 constraints

class Mimic3DatasetConstraint(Constraint):

    def __init__(self, net, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 0
        self.name = 'Mimic3'

    def params(self):
        return {'network_output': self.network_output}

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        x_batches = x_batches[0]
        x_out = self.net(x_batches)

        # Convert to probabilities 
        x_out = torch.sigmoid(x_out)
        
        # if we need targets - use this 
        targets = y_batches[0]

        rules = []

        # If min(systolic_blood_pressure) >= 180, output probability >= 0.2

        # param = 14 (max = 17)
        # time_bucket = 4 (max = 7)
        # param = 1 (max = 6)

        # 13 params before = 546 (13 * 7 * 6)
        # 3 time buckets before = 18 (3 * 6)
        # 546 + 18 -> 564 is the index of the 1st function of the 4th time bucket of the 14th parameter
        index_systolic_blood_pressure = 564
        min_normalized_systolic_blood_pressure = x_batches[:, index_systolic_blood_pressure]
        normalized_180 = 4.61654

        rule_normalized_systolic_blood_pressure = dl2.Implication(dl2.GEQ(min_normalized_systolic_blood_pressure, normalized_180), dl2.GEQ(x_out[:], 0.2))

        # min(Diastolic blood pressure) >= 120, output >= 0.2
        # 1 param before = 42 (1 * 7 * 6)
        # 3 time buckets = 18 (3 * 6)
        # 42 + 18 = 60 -> 60 is the index of the 1st function of the 4th time bucket of the 2nd parameter
        index_diastolic_blood_pressure = 60
        min_normalized_diastolic_blood_pressure = x_batches[:, index_diastolic_blood_pressure]
        
        # mean 60: 43.18149859748852
        # scale 60: 11.99327485987464
        # (120 - 43.18149859748852) / 11.99327485987464 = 6.40513140073
        normalized_120 = 6.40513140073

        rule_normalized_diastolic_blood_pressure = dl2.Implication(dl2.GEQ(min_normalized_diastolic_blood_pressure, normalized_120),
                        dl2.GEQ(x_out[:], 0.2))

        # Glasgow coma scale: risk should increase monotonically as total score decreases (3-8 is severe brain injury)
        # using max of 100% - maybe should use max of last 50% instead
        # 4 params before = 168 (4 * 7 * 6)
        # 0 times buckets before = 0
        # 1 function before = 1
        # 168 + 1 -> 169 is the index of the 1st function of the 1st time bucket of the 5th parameter
        index_glasgow_coma_total = 169
        max_glasgow_coma_total = x_batches[:, index_glasgow_coma_total] 

        # mean 169: 5.8098315578107576
        # scale 169: 0.7050565188848825

        # (3 - 5.8098315578107576) / 0.7050565188848825 = -3.98525718513
        # (8 - 5.8098315578107576) / 0.7050565188848825 = 3.10637287015
        # normalize these
        normalized_3 = -3.98525718513
        normalized_8 = 3.10637287015

        rule_glasgow_coma_total_3 = dl2.Implication(dl2.LEQ(max_glasgow_coma_total, normalized_3),
                        dl2.GEQ(x_out[:], 0.6))

        rule_glasgow_coma_total_8 = dl2.Implication(dl2.LEQ(max_glasgow_coma_total, normalized_8),
                        dl2.GEQ(x_out[:], 0.4))

        rules.append(rule_normalized_systolic_blood_pressure)
        rules.append(rule_normalized_diastolic_blood_pressure)  
        rules.append(rule_glasgow_coma_total_3)
        rules.append(rule_glasgow_coma_total_8)

        return dl2.And(rules)

