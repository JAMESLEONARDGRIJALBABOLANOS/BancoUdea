Feature: API de Clientes

  Background:
    * url 'http://localhost:8080/api'

  Scenario: Crear un nuevo cliente
    Given path 'customers'
    And request { firstName: 'Carlos', lastName: 'Gomez', accountNumber: 'KARATE-001', balance: 5000.0 }
    When method POST
    Then status 200
    And match response.firstName == 'Carlos'
    And match response.lastName == 'Gomez'
    And match response.accountNumber == 'KARATE-001'
    And match response.balance == 5000.0
    And match response.id == '#notnull'
    * def createdId = response.id

    # Limpiar: eliminar el cliente creado
    Given path 'customers/' + createdId
    When method DELETE
    Then status 204

  Scenario: Obtener todos los clientes
    Given path 'customers'
    When method GET
    Then status 200
    And match response == '#array'

  Scenario: Crear y obtener cliente por ID
    # Crear cliente
    Given path 'customers'
    And request { firstName: 'Ana', lastName: 'Torres', accountNumber: 'KARATE-002', balance: 2000.0 }
    When method POST
    Then status 200
    * def clienteId = response.id

    # Obtener por ID
    Given path 'customers/' + clienteId
    When method GET
    Then status 200
    And match response.firstName == 'Ana'
    And match response.accountNumber == 'KARATE-002'

    # Limpiar
    Given path 'customers/' + clienteId
    When method DELETE
    Then status 204

  Scenario: Actualizar un cliente existente
    # Crear cliente
    Given path 'customers'
    And request { firstName: 'Luis', lastName: 'Perez', accountNumber: 'KARATE-003', balance: 1000.0 }
    When method POST
    Then status 200
    * def clienteId = response.id

    # Actualizar
    Given path 'customers/' + clienteId
    And request { balance: 9999.0 }
    When method PUT
    Then status 200
    And match response.balance == 9999.0

    # Limpiar
    Given path 'customers/' + clienteId
    When method DELETE
    Then status 204

  Scenario: Eliminar un cliente
    # Crear cliente
    Given path 'customers'
    And request { firstName: 'Temp', lastName: 'User', accountNumber: 'KARATE-004', balance: 100.0 }
    When method POST
    Then status 200
    * def clienteId = response.id

    # Eliminar
    Given path 'customers/' + clienteId
    When method DELETE
    Then status 204
