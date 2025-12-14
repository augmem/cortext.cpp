# Entity Relationship Diagrams (Chen's notation)

This page is for Chen's [Entity Relationship](https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model) notation, which is commonly used in teaching. *See also [Information Engineering diagrams](ie-diagram).*

Entity Relationship (ER) diagrams are used to model databases at a conceptual level by describing entities, their attributes, and the relationships between them. In addition to basic relationships, PlantUML also supports subclasses and union types. This extended notation is sometimes referred to as Enhanced Entity Relationship (EER) or Extended Entity Relationship notation.

*\[Ref. [GH-945](https://github.com/plantuml/plantuml/issues/945) and [GH-1718](https://github.com/plantuml/plantuml/pull/1718)]*

## Minimal Example

### Vertical *(by default)*

<plantuml>
@startchen

entity Person {
}
entity Location {
}
relationship Birthplace {
}

Person -N- Birthplace
Birthplace -1- Location

@endchen </plantuml>

### Horizontal

<plantuml>
@startchen
left to right direction

entity Person {
}
entity Location {
}
relationship Birthplace {
}

Person -N- Birthplace
Birthplace -1- Location

@endchen </plantuml>

*\[Ref. [PR-1740](https://github.com/plantuml/plantuml/pull/1740)]*

## Entities and attributes

*Entities* correspond to the "things" in your model. These can have *attributes* that describe them and those attributes can be *composite* (having nested attributes).

<plantuml>
@startchen

entity DIRECTOR {
Name {
Fname
Lname
}
Born
Died
Age
}

entity MOVIE {
Title
Released
Code
}

@endchen </plantuml>

Attributes can be *keys*, meaning that their value is unique among entities of a given type, or they can be *derived*, meaning that their value is computed based on other attributes. Attributes may also be *multi-valued*, or have their *domain* (set of allowed values) defined.

<plantuml>
@startchen

entity DIRECTOR {
Number : INTEGER <<key>>
Name {
Fname : STRING
Lname : STRING
}
Born : DATE
Died : DATE
Age : INTEGER
}

entity CUSTOMER {
Number : INTEGER <<key>>
Bonus : REAL <<derived>>
Name : STRING <<multi>>
}

@endchen </plantuml>

## Relationships

*Relationships* describe how entities are related to each other. These can be *one-to-one*, *one-to-many*, or *many-to-many*. They can have *total participation* (mandatory) or *partial participation* (optional). Total participation is indicated using a double or thicker line. Relationships can also have attributes.

<plantuml>
@startchen

entity CUSTOMER {
Number <<key>>
Name
}

entity MOVIE {
Code <<key>>
}

relationship RENTED\_TO {
Date
}

RENTED\_TO =1= CUSTOMER
RENTED\_TO -N- MOVIE

@endchen </plantuml>

Relationships are not limited to two entities.

<plantuml>
@startchen

entity CUSTOMER {
Number <<key>>
Name
}

entity MOVIE {
Code <<key>>
}

entity INVOICE {
Number <<key>>
Amount
}

relationship RENTED\_TO {
Date
}

RENTED\_TO =1= CUSTOMER
RENTED\_TO -N- MOVIE
RENTED\_TO =1= INVOICE

relationship REFERENCES {
}

REFERENCES -1- MOVIE
REFERENCES -1- MOVIE

@endchen </plantuml>

### Structural constraints

The cardinality of relationships can also be expressed as a range.

<plantuml>
@startchen

entity CUSTOMER {
Number <<key>>
Name
}

entity MOVIE {
Code <<key>>
}

relationship RENTED\_TO {
Date
}

RENTED\_TO -(1,N)- CUSTOMER
RENTED\_TO -(0,1)- MOVIE

@endchen </plantuml>

## Identifying relationships

A *weak* entity does not have a key attribute that uniquely identifies each instance of that entity. Instead, it is identified by the combination of a *partial key* on the weak entity itself and the key of another entity, which it is related to via an *identifying relationship*. A weak entity must have total participation in its identifying relationship.

<plantuml>
@startchen

entity PARENT {
Number <<key>>
Name
}

entity CHILD <<weak>> {
Name <<key>>
Age
}

relationship PARENT\_OF <<identifying>> {
}

PARENT\_OF -1- PARENT
PARENT\_OF =N= CHILD

@endchen </plantuml>

## Aliases

Entities, attributes and relationships can be given aliases to make the diagram more readable.

<plantuml>
@startchen

entity "Customer" as CUSTOMER {
"customer number" as Number <<key>>
"member bonus" as Bonus <<derived>>
"first and last names" as Name <<multi>>
}

entity "Movie" as MOVIE {
"barcode" as Code
}

relationship "was-rented-to" as RENTED\_TO {
"date rented" as Date
}

RENTED\_TO -1- CUSTOMER
RENTED\_TO -N- MOVIE

@endchen </plantuml>

## Subclasses and categories

Entities can have *subclasses* and *superclasses*, much like in OOP, however a given subclass can have multiple superclasses. These are visually indicated using the subset symbol from set-theory.

<plantuml>
@startchen

entity CUSTOMER {
}

entity PARENT {
}

entity MEMBER {
}

CUSTOMER ->- PARENT
MEMBER -<- CUSTOMER

@endchen </plantuml>

We can show how the different subclasses of a given entity are related by combining the associations. They can be either *disjoint* (one at a time) or *overlapping* (multiple at the same time).

<plantuml>
@startchen

entity CUSTOMER {
}

entity PARENT {
}

entity MEMBER {
}

CUSTOMER ->- o { PARENT, MEMBER }

entity CHILD {
}

entity TODDLER {
}

entity PRIMARY\_AGE {
}

entity TEENAGER {
}

CHILD =>= d { TODDLER, PRIMARY\_AGE, TEENAGER }

@endchen </plantuml>

*Categories* or *union types* are similar to subclasses and can be used to group together multiple related entities.

<plantuml>
@startchen

entity CUSTOMER {
}

entity EMPLOYEE {
}

entity PERSON {
}

PERSON ->- U { CUSTOMER, EMPLOYEE }

@endchen </plantuml>

## Complex Example

<plantuml>
@startchen movies
<style>
.red {
BackGroundColor Red
FontColor White
}
.blue {
BackGroundColor Blue
FontColor White
}
</style>

entity "Director" as DIRECTOR {
"No." as Number <<key>>
Name {
Fname
Lname
}
Born : DATE
Died<<red>>
Age<<blue>>
}

entity "Customer" as CUSTOMER {
Number <<key>>
Bonus <<derived>>
Name <<multi>>
}

entity "Movie" as MOVIE {
Code
}

relationship "was-rented-to" as RENTED\_TO {
Date
}

RENTED\_TO -1- CUSTOMER
RENTED\_TO -N- MOVIE
RENTED\_TO -(N,M)- DIRECTOR

entity "Parent" as PARENT {
}

entity "Member" as MEMBER {
}

CUSTOMER ->- PARENT
MEMBER -<- CUSTOMER

entity "Kid" as CHILD <<weak>> {
Name <<key>>
}

relationship "is-parent-of" as PARENT\_OF <<identifying>> {
}

PARENT\_OF -1- PARENT
PARENT\_OF =N= CHILD

entity "Little Kid" as TODDLER {
FavoriteToy
}

entity "Primary-Aged Kid" as PRIMARY\_AGE {
FavoriteColor
}

entity "Teenager" as TEEN {
Hobby
}

CHILD =>= d { TODDLER, PRIMARY\_AGE, TEEN }

entity "Human" as PERSON {
}

PERSON ->- U { CUSTOMER, DIRECTOR }
@endchen </plantuml>
