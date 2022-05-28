from training.supervised.constraints import * # DL2 constraints

class Mimic3DatasetConstraint(Constraint):

    def __init__(self, net, eps1, eps2, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.eps1 = eps1
        self.eps2 = eps2
        self.use_cuda = use_cuda
        self.n_tvars = 2
        self.n_gvars = 0
        self.name = 'Mimic3'

    def params(self):
        return {'eps1': self.eps1, 'eps2': self.eps2}

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]

        # need to understand how to use the whole batch and compare it properly

        x_out1, x_out2 = self.net(x_batches[0]), self.net(x_batches[1])

        # Convert to probabilities
        x_out1 = torch.sigmoid(x_out1)
        x_out2 = torch.sigmoid(x_out2)

        rules = []

        # if min(systolic_blood_pressure) is higher than 180, die at least 20% of time

        # param = 14 (max = 17)
        # time_bucket = 4 (max = 7)
        # param = 1 (max = 6)

        # 13 params before = 546 (13 * 7 * 6)
        # 3 time buckets before = 18 (3 * 6)
        # 546 + 18 -> 564 is the index of the 1st function of the 4th time bucket of the 14th parameter
        index_systolic_blood_pressure = 564
        min_normalized_systolic_blood_pressure1 = x_batches[0][:, index_systolic_blood_pressure] 
        min_normalized_systolic_blood_pressure2 = x_batches[1][:, index_systolic_blood_pressure]

        normalized_systolic_blood_pressure_180 = 4.61654

        rule_normalized_systolic_blood_pressure1 = dl2.Implication(dl2.GEQ(min_normalized_systolic_blood_pressure1, normalized_systolic_blood_pressure_180),
                        dl2.GEQ(x_out1[:], 0.2))

        rule_normalized_systolic_blood_pressure2 = dl2.Implication(dl2.GEQ(min_normalized_systolic_blood_pressure2, normalized_systolic_blood_pressure_180),
                        dl2.GEQ(x_out2[:], 0.2))

        rules.append(rule_normalized_systolic_blood_pressure1)
        rules.append(rule_normalized_systolic_blood_pressure2)

        return dl2.And(rules)
