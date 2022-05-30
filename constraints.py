from training.supervised.constraints import * # DL2 constraints

# The headers from the in-hospital-mortality dataset
header = [
    'Capillary refill rate',
    'Diastolic blood pressure',
    'Fraction inspired oxygen',
    'Glascow coma scale eye opening',
    'Glascow coma scale motor response',
    'Glascow coma scale total',
    'Glascow coma scale verbal response',
    'Glucose',
    'Heart Rate',
    'Height',
    'Mean blood pressure',
    'Oxygen saturation',
    'Respiratory rate',
    'Systolic blood pressure',
    'Temperature',
    'Weight',
    'pH'
]

# Periods
subsequences = ['all', 'first10', 'first25', 'first50', 'last10', 'last25', 'last50']

# Features
sample_stats = ['min', 'max', 'mean', 'std', 'skew', 'len']

class Mimic3DatasetConstraint(Constraint):

    def __init__(self, net, scaler, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 0
        self.name = 'Mimic3'
        self.header = self.create_header()
        self.scaler = scaler


    def create_header(self):
        return [f"{head} - {seq} - {stat}" for head in header for seq in subsequences for stat in sample_stats]


    def normalize(self, value, idx):
        return self.scaler.transform([[value] * 714])[0, idx]


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

        # If min(systolic_blood_pressure) <= 80, output probability >= 0.2
        index_systolic_blood_pressure = self.header.index(f"Systolic blood pressure - last50 - min")
        min_normalized_systolic_blood_pressure = x_batches[:, index_systolic_blood_pressure]
        normalized_80 = self.normalize(80, index_systolic_blood_pressure)
        
        rule_normalized_systolic_blood_pressure = dl2.Implication(dl2.LEQ(min_normalized_systolic_blood_pressure, normalized_80), 
                        dl2.GEQ(x_out[:], 0.25))


        # min(Diastolic blood pressure) >= 120, output >= 0.2
        index_diastolic_blood_pressure =  self.header.index(f"Diastolic blood pressure - last50 - min")
        min_normalized_diastolic_blood_pressure = x_batches[:, index_diastolic_blood_pressure]
        normalized_120 = self.normalize(60, index_diastolic_blood_pressure)

        rule_normalized_diastolic_blood_pressure = dl2.Implication(dl2.GEQ(min_normalized_diastolic_blood_pressure, normalized_120),
                        dl2.GEQ(x_out[:], 0.2))


        # min(Glascow coma scale total) <= 3, output >= 0.6
        # min(Glascow coma scale total) <= 8, output >= 0.4
        index_glascow_coma_total = self.header.index(f"Glascow coma scale total - last50 - min")
        min_glascow_coma_total = x_batches[:, index_glascow_coma_total] 
        normalized_3 = self.normalize(3, index_glascow_coma_total)
        normalized_8 = self.normalize(8, index_glascow_coma_total)

        rule_glascow_coma_total_3 = dl2.Implication(dl2.LEQ(min_glascow_coma_total, normalized_3),
                        dl2.GEQ(x_out[:], 0.6))
        rule_glascow_coma_total_8 = dl2.Implication(dl2.LEQ(min_glascow_coma_total, normalized_8),
                        dl2.GEQ(x_out[:], 0.4))

        # Append rules
        rules.append(rule_normalized_systolic_blood_pressure)
        rules.append(rule_normalized_diastolic_blood_pressure)  
        rules.append(rule_glascow_coma_total_3)
        rules.append(rule_glascow_coma_total_8)

        return dl2.And(rules)

