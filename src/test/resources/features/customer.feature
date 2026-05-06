Feature: Gestion de clientes

  Scenario: Crear un nuevo cliente exitosamente
    Given un cliente con nombre "Juan", apellido "Perez", cuenta "ACC-001" y saldo 1000.0
    When se crea el cliente
    Then el cliente es retornado con los mismos datos

  Scenario: Obtener un cliente por ID existente
    Given un cliente guardado con ID 1
    When se busca el cliente con ID 1
    Then se retorna el cliente correctamente

  Scenario: Obtener un cliente por ID inexistente
    Given que no existe un cliente con ID 99
    When se busca el cliente con ID 99
    Then se lanza una excepcion con mensaje "Cliente no encontrado con id: 99"

  Scenario: Eliminar un cliente existente
    Given un cliente guardado con ID 1
    When se elimina el cliente con ID 1
    Then el cliente es eliminado sin errores

  Scenario: Eliminar un cliente inexistente
    Given que no existe un cliente con ID 99
    When se elimina el cliente con ID 99
    Then se lanza una excepcion con mensaje "Cliente no encontrado con id: 99"
