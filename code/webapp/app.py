from flask import Flask, render_template, request , redirect, session
from flask_sqlalchemy import SQLAlchemy
from model import prediction , model
import pandas as pd
import json
df = pd.read_csv("data.csv")
df["directions"] = df["directions"].apply(json.loads)
app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///test.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = '378934789'

db = SQLAlchemy(app)

class Ingredients(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ingredient = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<Ingredients {self.id}>'
    




@app.route("/",methods=['POST','GET'])
def index():
    title = []
    directions = []
    link = []
    show_predictions = session.get('show_predictions', False)
    if request.method == 'POST':
        ingreds= request.form['test']
        if ingreds.replace(" ","") ==  "" or ingreds.replace(" ","") in [x.ingredient.replace(" ","") for x in Ingredients.query.order_by(Ingredients.id).all()]:
            return redirect('/')
        
        new_ingr = Ingredients(ingredient=ingreds)
        try:
            db.session.add(new_ingr)
            db.session.commit()
            return redirect('/')
        except:
            return "Eroor 404"
    else :
        ingreds = Ingredients.query.order_by(Ingredients.id).all()

        if len(ingreds) > 0 and show_predictions:

            ingredient_list = [i.ingredient for i in ingreds]
            id = prediction(ingredient_list)

            title = [df.iloc[i, 2] for i in id]
            print(title)
            directions = [df.iloc[i, 4] for i in id]
            print(directions)
            link = [df.iloc[i, 5] for i in id]
            print(link)

        return render_template("index.html", ingreds=ingreds, title=title, directions=directions, link=link, show_predictions=show_predictions)


@app.route("/delete/<int:id>")
def delete(id):
    ingred_delete = Ingredients.query.get_or_404(id)
    try:
        db.session.delete(ingred_delete)
        db.session.commit()
        return redirect("/")
    except:
        return "an eroor has happened impossible to delete ur ingred"

@app.route("/clear_all")
def clear_all():
    try:
        Ingredients.query.delete()
        db.session.commit()  
        session['show_predictions'] = False
        return redirect("/")
    except:
        return "An error occurred while clearing the database."
    
@app.route("/suggest")
def suggest():
    session['show_predictions'] = True
    return redirect('/')
    

@app.route("/clear_prediction")
def clear_prediction():
    session['show_predictions'] = False
    return redirect('/')


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
