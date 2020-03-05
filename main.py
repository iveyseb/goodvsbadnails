import web
#import sys,os
import json
import rf as model

urls = ('/nails', 'model')
app = web.application(urls, globals())


class nails:
    def GET(self):
        web.header('Content-Type', 'application/json')
        user_data = web.input()
        comment= user_data.message
#        senderid= user_data.senderid
#        senderName= user_data.senderName
            
        pred = model.process_test_data(img)
        if pred==1: 
            str_label='good'
        else:
            str_label='bad'
        y.imshow(img, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        plt.show()  
        return json.dumps(str_label)



if __name__ == "__main__":
    app.run()