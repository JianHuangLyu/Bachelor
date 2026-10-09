import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.roi_pooling.modules import roi_pooling
from torchvision import transforms
from PIL import Image


class attention_layer(nn.Module):
    def __init__(self, ev_dim):
        super(attention_layer, self).__init__()
        self.fc_out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(ev_dim,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,2)
        )
        self.fc1 = nn.Sequential(nn.Linear(ev_dim,ev_dim))

    def forward(self, input):
        # x = nn.functional.interpolate(input,scale_factor=2)
        # x_conv_feature = self.conv(x)
        x = self.fc1(input)
        att_probs = torch.sigmoid(x)
        ev = torch.mul(input,att_probs)
        x = ev.view(ev.size(0), -1)

        x_out = self.fc_out(x)
        return x_out, ev


class FeatRef(nn.Module):
    def __init__(self,  num_class, args):
        self.args = args

        super(FeatRef, self).__init__()

        self.att = nn.ModuleList([attention_layer( self.args.ev_dim) for i in range(num_class)])
        ############################################
        in_chanel = 3
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_chanel, 6, kernel_size= (1,1), stride=1, padding=0),
            nn.BatchNorm2d(6),

            nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_chanel, 6, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(6),

            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(6),

            nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_chanel, 6, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(6),

            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(6),

            nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_chanel, 6, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(6),

            nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size= (1,1), stride=1, padding=0),
            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True))
        self.conv2_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(24, 16, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.apedx = nn.Sequential(
                                    nn.Linear(self.args.ev_dim, num_class))
        self.fc = nn.Sequential( nn.Linear(8472,  self.args.ev_dim),
                                 nn.Dropout(0.3),
                                 nn.ReLU(inplace=True))
                                 # nn.BatchNorm1d(self.args.ev_dim))


        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(inplace=True)
            )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(inplace=True)
            )
        self.fc1 = nn.Linear(880, 100)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def inception_seq_x(self, x):
        x11 = self.conv1_1(x)
        x12 = self.conv1_2(x)
        x13 = self.conv1_3(x)
        x14 = self.conv1_4(x)
        x1_con1 = torch.cat([x11,x12,x13,x14], 1)
        x1_con1 = torch.max_pool2d(x1_con1, kernel_size=(2, 2), stride=(2, 2))
        y11 = self.conv2_1(x1_con1)
        y12 = self.conv2_2(x1_con1)
        y13 = self.conv2_3(x1_con1)
        y14 = self.conv2_4(x1_con1)
        y1_con2 = torch.cat([y11, y12, y13, y14], 1)
        out = torch.max_pool2d(y1_con2, kernel_size=(2, 2),stride=(2,2))
        # out = out1.view(-1, self.num_flat_features(out1))
        return out
    def inception_seq_y(self, x):
        x11 = self.conv1_1(x)
        x12 = self.conv1_2(x)
        x13 = self.conv1_3(x)
        x14 = self.conv1_4(x)
        x1_con1 = torch.cat([x11,x12,x13,x14], 1)
        x1_con1 = torch.max_pool2d(x1_con1, kernel_size=(2, 2), stride=(2, 2))
        y11 = self.conv2_1(x1_con1)
        y12 = self.conv2_2(x1_con1)
        y13 = self.conv2_3(x1_con1)
        y14 = self.conv2_4(x1_con1)
        y1_con2 = torch.cat([y11, y12, y13, y14], 1)
        out = torch.max_pool2d(y1_con2, kernel_size=(2, 2),stride=(2,2))
        return out
    
    def cnn2d(self,x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = x.view(-1, 880)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x
    

    def forward(self, inputx):
        # print(input_x_rois.size())
        sub_out_list = []        
        B, T, C, W, H = inputx.size()
        # representation
        x1 = self.inception_seq_x(inputx[:, 0])
        x2 = self.inception_seq_y(inputx[:, 1])
        
        
        x3 = inputx[:, 2:]
        Bx3, Tx3, Cx3, Wx3, Hx3 = x3.size()
        for i in range(Tx3):
            if i == 0:
                conv2d_x3=self.cnn2d(x3[:,i])
            else:
                conv2d_x3=torch.cat([conv2d_x3, self.cnn2d(x3[:,i])], dim=1)
        
        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.size(0), -1)
        conv2d_x3 = conv2d_x3.view(x.size(0),-1)
        x = torch.cat([x, conv2d_x3], dim=1)
        x = self.fc(x)
        for i in range(self.args.class_num):
            ml_out, feature = self.att[i](x)
            sub_out_list.append(ml_out)
            if i == 0:
                feat = feature
            else:
                feat = torch.add(feat, feature)
        feat = feat.view(feat.size(0), -1)
        x = self.apedx(feat)
        return x
