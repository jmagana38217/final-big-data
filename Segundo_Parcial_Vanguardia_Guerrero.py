#Resolver el ejercicio de la vanguardia guerrero Ejercicio_Refactorización_Equipo.docx
#Ejercicio

list_get_only_username = []

def process_split(emails_list):
    emails_list = None
    list_get_only_username = []
    try:
        
        for i in emails_list: # el iterador "i", recorre la lista "d", ahora data.
            if "@" in i: #Este identifica si es un email, lo que obitene en d, ahora data.
                list_get_only_username.append(i.lower().strip()) #Si el valor existe, lo convierte a minúsculas.
            
            else:
                print("invalid:", i) #Si no encuentra un valor de email, imprime el índice "invalid" y el valor en la lista data que no contiene un email.
        print(list_get_only_username)
    except Exception as e:
        print("Lista vacía")
    for x in list_get_only_username:
        username = x.split("@")[0] #Esta función separa el elemento de la lista en dos, y usa el "@" como un separador.
        #print("username:", username) #Imprime la parte anterior al "@", que es el nombre del usuario.
    return list_get_only_username

emails_list = ["  ALICE@MAIL.COM", "bob@gmail.com", "invalid_email", " charlie@live.com "]

def process_getusers(list_get_only_username):
    list_get_only_username = []
    for x in list_get_only_username:
        username = x.split("@")[0] #Esta función separa el elemento de la lista en dos, y usa el "@" como un separador.
        print("username:", username) #Imprime la parte anterior al "@", que es el nombre del usuario.
    return list_get_only_username
process_split(emails_list)
process_getusers(list_get_only_username)

